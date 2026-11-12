"""LLM 백엔드 인터페이스. 오케스트레이터는 이 인터페이스만 안다.

구현 예정: mock (지금), gemini_cli / gemini_api (공식 경로), gemini_web (브라우저 자동화, 선택).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..roles import Role


class BackendError(Exception):
    """백엔드 일반 오류."""


class UsageLimitReached(BackendError):
    """사용량 한도 도달 (웹 배너, API 429 등). 오케스트레이터는 저장 후 안전 종료한다."""


class ResponseTimeout(BackendError):
    """응답 완료를 제한 시간 안에 감지하지 못함."""


class ChatBackend(ABC):
    """역할별 대화 채널을 제공하는 백엔드.

    stateful=True 면 백엔드가 역할별 대화 기록을 스스로 유지한다 (웹 탭, 세션형 API).
    False 면 매 호출이 독립적이므로 프로토콜이 필요한 맥락을 프롬프트에 담아 보낸다.
    """

    stateful: bool = True

    @abstractmethod
    def open(self, roles: tuple[Role, ...]) -> None:
        """역할별 채널 준비 (탭 열기, 세션 생성 등)."""

    @abstractmethod
    def send(self, role: Role, prompt: str) -> str:
        """prompt 를 role 채널로 보내고 완성된 응답 텍스트를 반환한다.

        Raises:
            UsageLimitReached, ResponseTimeout, BackendError
        """

    def close(self) -> None:  # noqa: B027 - 선택적 오버라이드
        """자원 정리. 기본은 아무것도 안 함."""
