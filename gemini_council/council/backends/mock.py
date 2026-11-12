"""가짜 백엔드: 외부 연결 없이 오케스트레이터/프로토콜을 검증한다."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..roles import Role
from .base import BackendError, ChatBackend, ResponseTimeout, UsageLimitReached

_STANCES = ("동의", "부분 동의", "반박")


@dataclass
class MockBackend(ChatBackend):
    """결정적(시드 고정) 가짜 응답기.

    Attributes:
        stateful: 프로토콜의 stateful/stateless 경로를 둘 다 시험하기 위해 선택 가능.
        seed: 응답 내용 난수 시드 (전역 random 대신 인스턴스 Random 사용).
        end_at_round: 사회자가 이 라운드에서 종료 토큰을 출력. None 이면 출력 안 함.
        end_token: 종료 토큰 문자열 (config 와 같게 맞출 것).
        fail_at_turn: 이 턴 번호(1부터)에서 fail_with 예외를 발생시킴. None 이면 안 함.
        fail_with: "limit" | "timeout" | "error" | "crash"
    """

    stateful: bool = True
    seed: int = 0
    end_at_round: int | None = None
    end_token: str = "[DISCUSSION_COMPLETE]"
    fail_at_turn: int | None = None
    fail_with: str = "limit"
    prompts: list[tuple[str, str]] = field(default_factory=list, init=False)
    _calls: dict[str, int] = field(default_factory=dict, init=False)
    _rng: random.Random = field(init=False)
    opened: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def open(self, roles: tuple[Role, ...]) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, role: Role, prompt: str) -> str:
        if not self.opened:
            raise BackendError("open() 전에 send() 호출됨")
        self.prompts.append((role.key, prompt))
        turn = len(self.prompts)
        if self.fail_at_turn is not None and turn == self.fail_at_turn:
            if self.fail_with == "limit":
                raise UsageLimitReached("모의 사용량 한도 도달")
            if self.fail_with == "timeout":
                raise ResponseTimeout("모의 응답 타임아웃")
            if self.fail_with == "error":
                raise BackendError("모의 백엔드 오류")
            raise RuntimeError("모의 예기치 못한 오류")  # 'crash': 백엔드 예외가 아닌 버그 흉내

        if "[토론 종료 — 초록 작성]" in prompt:
            return "**배경·주제:** 모의 주제\n**핵심 쟁점:** 모의 쟁점\n**합의된 결론:** 모의 결론\n" \
                   "**미해결·검증 필요:** 없음\n**다음 단계:** 모의 단계"
        n = self._calls.get(role.key, 0) + 1  # 이 역할의 n번째 발언 = 라운드 번호
        self._calls[role.key] = n
        if role.is_moderator:
            body = (f"(1) 라운드 요약: 라운드 {n} 쟁점 정리 #{self._rng.randint(100, 999)}\n"
                    f"(2) 남은 쟁점: {self._rng.randint(0, 3)}개\n(3) 다음 질문: 비용 상한은?")
            if self.end_at_round is not None and n >= self.end_at_round:
                body += f"\n최종 결론: 수렴함\n{self.end_token}"
            return body
        return (f"(1) 입장: {self._rng.choice(_STANCES)}\n"
                f"(2) 근거: {role.name} 라운드 {n} 근거 수치 {self._rng.uniform(0, 100):.1f}\n"
                f"(3) 제안: 수정안 #{self._rng.randint(100, 999)}")
