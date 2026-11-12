"""토론 프로토콜의 순수 로직 (브라우저/네트워크 의존 없음).

핵심 설계: 매 턴 전체 대화를 다시 보내지 않는다.
- stateful 백엔드 (웹 탭처럼 대화 기록을 스스로 기억): 그 역할이 마지막으로 말한 뒤
  나온 '새 발언(델타)'만 보낸다. → 턴당 입력 길이가 라운드 수와 무관하게 거의 일정.
- stateless 백엔드 (API/CLI 1회 호출): seed + 사회자의 최신 요약 + 그 요약 이후 발언만 보낸다.
  → 사회자 요약이 '압축된 기억' 역할을 하므로 입력 길이가 역시 상한을 가진다.
전체 대화를 매번 보내면 총 입력량이 O(N^2) (N=턴 수)로 늘어 사용량 한도를 빠르게 소모한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CouncilConfig
from .roles import Role, build_seed_prompt


@dataclass(frozen=True)
class Utterance:
    """발언 1개. round 는 1부터 시작."""

    round: int
    speaker_key: str
    speaker_name: str
    text: str

    @property
    def chars(self) -> int:
        """공백 포함 문자 수."""
        return len(self.text)


@dataclass
class DiscussionState:
    """누적 발언 버퍼 (중앙 공유 메모리).

    last_turn_end[key] = 그 역할이 마지막으로 발언한 직후의 utterances 인덱스.
    아직 발언하지 않은 역할은 키가 없으며 '처음부터 전부'가 델타가 된다.
    """

    utterances: list[Utterance] = field(default_factory=list)
    last_turn_end: dict[str, int] = field(default_factory=dict)

    def record(self, utt: Utterance) -> None:
        self.utterances.append(utt)
        self.last_turn_end[utt.speaker_key] = len(self.utterances)

    def has_spoken(self, key: str) -> bool:
        return key in self.last_turn_end

    def delta_for(self, key: str) -> list[Utterance]:
        """key 역할이 마지막 발언 이후 아직 못 본 다른 참가자의 발언 (시간 순)."""
        start = self.last_turn_end.get(key, 0)
        return [u for u in self.utterances[start:] if u.speaker_key != key]

    def latest_summary_index(self, moderator_key: str) -> int | None:
        """가장 최근 사회자 발언의 인덱스. 없으면 None."""
        for i in range(len(self.utterances) - 1, -1, -1):
            if self.utterances[i].speaker_key == moderator_key:
                return i
        return None


def _format_utterances(utts: list[Utterance]) -> str:
    if not utts:
        return "(없음)"
    return "\n\n".join(f"### [라운드 {u.round}] {u.speaker_name}\n{u.text.strip()}" for u in utts)


def _moderator_key(roles: tuple[Role, ...]) -> str:
    return next(r.key for r in roles if r.is_moderator)


def build_turn_prompt(
    role: Role,
    roles: tuple[Role, ...],
    state: DiscussionState,
    round_no: int,
    topic: str,
    cfg: CouncilConfig,
    stateful: bool,
) -> str:
    """이번 턴에 role 에게 보낼 메시지를 만든다.

    Args:
        round_no: 현재 라운드 (1..cfg.max_rounds).
        stateful: 백엔드가 대화 기록을 스스로 유지하는지 여부.
    Returns:
        백엔드 입력창에 그대로 넣을 문자열.
    """
    # 전체 라운드 수(예산)는 알려주지 않는다 → 마지막 라운드에 맞춘 억지 수렴 방지 (roles.build_seed_prompt 참고)
    header = f"[현재] 라운드 {round_no}, {role.name}의 차례입니다."
    footer = "위 내용을 바탕으로 당신의 역할과 규칙에 맞게 발언하세요."

    if stateful:
        parts: list[str] = []
        if not state.has_spoken(role.key):
            parts.append(build_seed_prompt(role, roles, topic, cfg))
        parts += [header, "[지난 발언 이후 새 발언]", _format_utterances(state.delta_for(role.key)), footer]
        return "\n\n".join(parts)

    # stateless: seed + (최신 사회자 요약) + 요약 이후 발언
    mod_key = _moderator_key(roles)
    idx = state.latest_summary_index(mod_key)
    if idx is None:
        summary_block = "(아직 없음)"
        recent = state.utterances
    else:
        summary_block = _format_utterances([state.utterances[idx]])
        recent = state.utterances[idx + 1:]
    return "\n\n".join([
        build_seed_prompt(role, roles, topic, cfg),
        header,
        "[사회자의 최신 요약]",
        summary_block,
        "[요약 이후 발언]",
        _format_utterances(recent),
        footer,
    ])


# 마크다운 강조(**, __, `)로 감싼 토큰도 인정한다. 예: **[DISCUSSION_COMPLETE]**
_DECORATION = re.compile(r"^[\s*_`>]+|[\s*_`.]+$")


def is_termination(role: Role, reply: str, cfg: CouncilConfig) -> bool:
    """사회자의 응답 '마지막 비어 있지 않은 줄'이 종료 토큰 단독일 때만 True.

    본문 중간의 인용이나 다른 역할의 발언은 종료로 보지 않는다 (오탐 방지).
    """
    if not role.is_moderator:
        return False
    lines = [ln for ln in reply.splitlines() if ln.strip()]
    if not lines:
        return False
    return _DECORATION.sub("", lines[-1]) == cfg.end_token
