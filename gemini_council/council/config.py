"""토론 실행 파라미터 (모든 매직넘버는 여기에 모은다)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CouncilConfig:
    """토론 1회 실행 설정.

    Attributes:
        max_rounds: 최대 라운드 수. 1 라운드 = 모든 역할이 1번씩 발언 (기본 4턴).
        char_limit: 발언 1회 권장 상한 [문자, 공백 포함]. 프롬프트에 지시하고, 넘으면 경고만 기록한다.
        end_token: 사회자가 토론 종료 시 마지막 줄에 단독으로 출력하는 문자열.
        log_dir: 회의록 저장 폴더 (없으면 자동 생성).
    """

    max_rounds: int = 5
    char_limit: int = 800
    end_token: str = "[DISCUSSION_COMPLETE]"
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds 는 1 이상이어야 함: {self.max_rounds}")
        if self.char_limit < 50:
            raise ValueError(f"char_limit 이 너무 작음: {self.char_limit}")
        if not self.end_token.strip():
            raise ValueError("end_token 이 비어 있음")
