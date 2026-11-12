"""토론 메인 루프. 백엔드 종류와 무관하게 동작한다.

종료 경로는 모두 회의록 저장을 거친다 (정상 종료, 한도, 타임아웃, 사용자 중단, 예기치 못한 오류).
예기치 못한 오류는 저장 후 그대로 다시 던진다 (에러를 삼켜서 '통과'시키지 않음).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from .backends.base import BackendError, ChatBackend, ResponseTimeout, UsageLimitReached
from .config import CouncilConfig
from .protocol import DiscussionState, Utterance, build_turn_prompt, is_termination
from .roles import DEFAULT_ROLES, Role, validate_roles
from .transcript import render_markdown, save_transcript


class StopReason(str, Enum):
    COMPLETED = "사회자 종료 선언"
    MAX_ROUNDS = "최대 라운드 도달"
    USAGE_LIMIT = "사용량 한도 감지"
    TIMEOUT = "응답 타임아웃"
    BACKEND_ERROR = "백엔드 오류"
    USER_STOP = "사용자 중단"
    CRASH = "예기치 못한 오류"


@dataclass(frozen=True)
class RunResult:
    reason: StopReason
    rounds_done: int  # 모든 역할이 발언을 마친 라운드 수
    utterances: tuple[Utterance, ...]
    log_path: Path
    detail: str = ""


EventCallback = Callable[[str], None]


class Orchestrator:
    """토론 1회 실행기.

    Args:
        backend: ChatBackend 구현.
        topic: 토론 주제 (빈 문자열 불가).
        cfg: 실행 설정.
        roles: 발언 순서대로의 역할 (사회자가 마지막).
        on_event: 진행 로그 콜백 (CLI print, GUI queue 등).
        stop_event: set 되면 다음 턴 시작 전에 중단·저장.
        resume_event: clear 상태면 다음 턴 시작 전에 대기 (일시정지). 기본은 set(진행).
        clock: 현재 시각 함수 (테스트에서 고정하기 위함).
    """

    def __init__(
        self,
        backend: ChatBackend,
        topic: str,
        cfg: CouncilConfig,
        roles: tuple[Role, ...] = DEFAULT_ROLES,
        on_event: EventCallback | None = None,
        stop_event: threading.Event | None = None,
        resume_event: threading.Event | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        if not topic.strip():
            raise ValueError("토론 주제가 비어 있음")
        validate_roles(roles)
        self.backend = backend
        self.topic = topic.strip()
        self.cfg = cfg
        self.roles = roles
        self.on_event = on_event or (lambda msg: None)
        self.stop_event = stop_event or threading.Event()
        self.resume_event = resume_event or threading.Event()
        if resume_event is None:
            self.resume_event.set()
        self.clock = clock
        self.state = DiscussionState()

    def _wait_if_paused(self) -> None:
        if not self.resume_event.is_set():
            self.on_event("일시정지됨")
            while not self.resume_event.wait(timeout=0.2):
                if self.stop_event.is_set():
                    return
            self.on_event("재개")

    def _loop(self) -> StopReason:
        for r in range(1, self.cfg.max_rounds + 1):
            for role in self.roles:
                self._wait_if_paused()
                if self.stop_event.is_set():
                    return StopReason.USER_STOP
                prompt = build_turn_prompt(role, self.roles, self.state, r, self.topic,
                                           self.cfg, self.backend.stateful)
                self.on_event(f"[R{r}] {role.name} 발언 요청 (입력 {len(prompt)}자)")
                reply = self.backend.send(role, prompt)
                utt = Utterance(r, role.key, role.name, reply)
                self.state.record(utt)
                warn = " ⚠ 길이 초과" if utt.chars > self.cfg.char_limit else ""
                self.on_event(f"[R{r}] {role.name} 응답 {utt.chars}자{warn}")
                if is_termination(role, reply, self.cfg):
                    return StopReason.COMPLETED
        return StopReason.MAX_ROUNDS

    def run(self) -> RunResult:
        started = self.clock()
        reason, detail = StopReason.CRASH, ""
        crash: BaseException | None = None
        try:
            self.backend.open(self.roles)
            reason = self._loop()
        except UsageLimitReached as e:
            reason, detail = StopReason.USAGE_LIMIT, str(e)
        except ResponseTimeout as e:
            reason, detail = StopReason.TIMEOUT, str(e)
        except BackendError as e:
            reason, detail = StopReason.BACKEND_ERROR, str(e)
        except KeyboardInterrupt:
            reason, detail = StopReason.USER_STOP, "KeyboardInterrupt"
        except BaseException as e:  # 저장 후 다시 던진다
            reason, detail, crash = StopReason.CRASH, f"{type(e).__name__}: {e}", e
        finally:
            try:
                self.backend.close()
            finally:
                finished = self.clock()
                md = render_markdown(self.topic, self.state.utterances, self.cfg,
                                     reason.value, started, finished, detail,
                                     appendix=self.backend.usage_report())
                log_path = save_transcript(md, self.cfg.log_dir, finished)
                self.on_event(f"종료: {reason.value}. 회의록 저장: {log_path}")
        if crash is not None:
            raise crash
        return RunResult(reason, self._complete_rounds(), tuple(self.state.utterances), log_path, detail)

    def _complete_rounds(self) -> int:
        """모든 역할이 발언을 마친 라운드 수 (중간에 끊긴 라운드는 제외)."""
        return len(self.state.utterances) // len(self.roles)
