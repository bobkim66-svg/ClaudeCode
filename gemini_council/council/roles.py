"""역할(페르소나) 정의와 첫 프롬프트(seed) 생성."""
from __future__ import annotations

from dataclasses import dataclass

from .config import CouncilConfig


@dataclass(frozen=True)
class Role:
    """토론 참가자 1명.

    Attributes:
        key: 내부 식별자 (영문, 고유).
        name: 회의록/프롬프트에 쓰는 표시 이름.
        mission: 역할 설명 (seed 프롬프트에 들어감).
        model_hint: 백엔드가 고를 모델 종류 힌트. "fast" | "thinking".
        is_moderator: True 면 라운드 요약과 종료 판단을 맡는다. 종료 토큰은 이 역할에게만 알려준다.
    """

    key: str
    name: str
    mission: str
    model_hint: str = "fast"
    is_moderator: bool = False


DEFAULT_ROLES: tuple[Role, ...] = (
    Role(
        key="proposer",
        name="제안자",
        mission="주제에 대한 초기 설계안과 구체적 구현 방안을 제시하고, 이후 라운드에서는 지적을 반영해 안을 갱신한다.",
    ),
    Role(
        key="critic",
        name="비평가",
        mission="제안된 안의 논리적 허점, 예외 상황, 물리적·비용적 한계를 근거와 함께 엄격하게 검증한다.",
        model_hint="thinking",
    ),
    Role(
        key="pragmatist",
        name="대안 설계자",
        mission="비평가의 지적을 반영해 현실적으로 실행 가능한 수정안과 대안을 도출한다.",
    ),
    Role(
        key="moderator",
        name="사회자",
        mission="이번 라운드의 핵심 쟁점을 5줄 이내로 요약하고, 남은 쟁점과 토론 수렴 여부를 판단한다.",
        is_moderator=True,
    ),
)


def validate_roles(roles: tuple[Role, ...]) -> None:
    """역할 목록의 불변 조건 검사: 비어 있지 않음, key 고유, 사회자 정확히 1명이며 마지막 순서."""
    if not roles:
        raise ValueError("역할이 하나도 없음")
    keys = [r.key for r in roles]
    if len(set(keys)) != len(keys):
        raise ValueError(f"역할 key 중복: {keys}")
    mods = [r for r in roles if r.is_moderator]
    if len(mods) != 1:
        raise ValueError(f"사회자는 정확히 1명이어야 함 (현재 {len(mods)}명)")
    if not roles[-1].is_moderator:
        raise ValueError("사회자는 라운드의 마지막 발언자여야 함 (요약·종료 판단을 위해)")


def build_seed_prompt(role: Role, roles: tuple[Role, ...], topic: str, cfg: CouncilConfig) -> str:
    """역할에게 처음 한 번 주는 규칙 + 페르소나 프롬프트.

    종료 토큰은 사회자 seed 에만 넣는다. 다른 참가자가 토큰을 인용해 토론이 잘못 끝나는 것을 막기 위함.
    """
    members = ", ".join(r.name for r in roles)
    lines = [
        "당신은 여러 AI가 참여하는 구조화된 토론의 참가자입니다.",
        f"[토론 주제] {topic}",
        f"[참가자 발언 순서] {members}",
        f"[당신의 역할] {role.name}: {role.mission}",
        "",
        "[토론 규칙]",
        "1. 인사말, 칭찬, 감사, 감정적 동조 표현을 쓰지 않는다.",
    ]
    if role.is_moderator:
        lines += [
            "2. 형식: (1) 라운드 요약: 핵심 쟁점 5줄 이내 (2) 남은 쟁점 (3) 다음 라운드에 다룰 질문",
            f"3. 공백 포함 {cfg.char_limit}자 이내로 쓴다.",
            "4. 새로운 반론이나 미해결 쟁점이 더 이상 없다고 판단되면, 최종 결론을 정리한 뒤 "
            f"답변의 맨 마지막 줄에 {cfg.end_token} 만 단독으로 출력한다. "
            "그 외의 경우에는 이 문자열을 절대 쓰지 않는다.",
        ]
    else:
        lines += [
            "2. 모든 발언은 3단 구조로만 쓴다: (1) 입장: 동의/부분 동의/반박 중 하나 "
            "(2) 근거: 구체적 근거나 수치 (3) 제안: 수정안 또는 다음 질문",
            f"3. 공백 포함 {cfg.char_limit}자 이내, 불릿 5개 이내로 쓴다.",
            "4. 다른 참가자의 발언을 그대로 되풀이하지 않는다.",
        ]
    return "\n".join(lines)
