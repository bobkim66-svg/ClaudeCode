"""토론 메인 루프. 백엔드 종류와 무관하게 동작한다.

종료 경로는 모두 회의록 저장을 거친다 (정상 종료, 한도, 타임아웃, 사용자 중단, 예기치 못한 오류).
예기치 못한 오류는 저장 후 그대로 다시 던진다 (에러를 삼켜서 '통과'시키지 않음).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from .backends.base import BackendError, ChatBackend, ResponseTimeout, UsageLimitReached, UserAbort
from .config import CouncilConfig
from .protocol import DiscussionState, Utterance, build_abstract_prompt, build_turn_prompt, is_termination
from .roles import DEFAULT_ROLES, Role, validate_roles
from .session import (STATE_VERSION, STATUS_RUNNING, SessionError, save_state, state_path_for,
                      utterances_from_state, utterances_to_state, write_atomic)
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
        resume: session.load_state 로 읽은 진행 상태. 주어지면 그 토론을 이어서 진행하고 같은 파일에 기록한다.
        resume_json: resume 상태 파일 경로 (회의록·상태 파일이 있는 폴더를 정함).
        make_abstract: 정상 종료·최대 라운드 도달 시 사회자에게 토론 전체 초록을 한 번 더 요청해 회의록 맨 위에 넣는다.

    발언 하나가 끝날 때마다 회의록(.md)과 진행 상태(.json)를 자동 저장한다.
    창을 닫거나 PC 가 꺼져도 마지막으로 끝난 발언까지는 남고, --resume 으로 이어 할 수 있다.
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
        resume: dict | None = None,
        resume_json: Path | None = None,
        make_abstract: bool = True,
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
        self._resume = resume
        self.make_abstract = make_abstract
        self.abstract = ""
        self.abstract_note = ""
        self._needs_recap: set[str] = set()
        self._resumes: list[str] = []
        self._started: datetime | None = None
        self._md_path: Path | None = None
        if resume is not None:
            if resume_json is None:
                raise ValueError("resume 에는 resume_json 경로가 필요함")
            self._load_resume(resume, resume_json)

    # ------------------------------------------------------------------ 이어 하기 / 자동 저장

    def _load_resume(self, data: dict, json_path: Path) -> None:
        utts = utterances_from_state(data)
        n_roles = len(self.roles)
        for k, u in enumerate(utts):  # 불변 조건: 저장된 발언 순서가 역할 순서와 일치
            expected = self.roles[k % n_roles]
            if u.speaker_key != expected.key or u.round != k // n_roles + 1:
                raise SessionError(f"저장된 발언 순서가 현재 역할 구성과 맞지 않음 ({k + 1}번째 발언: {u.speaker_name})")
        if data["status"] == "COMPLETED":
            raise SessionError("이미 사회자가 종료를 선언한 토론이라 이어 할 것이 없음")
        if len(utts) >= self.cfg.max_rounds * n_roles:
            raise SessionError(f"이미 최대 라운드({self.cfg.max_rounds})까지 진행됨. 더 하려면 더 큰 --rounds 를 지정")
        for u in utts:
            self.state.record(u)
        self._needs_recap = {r.key for r in self.roles}  # 탭이 이전 대화를 잃었다고 가정
        self._resumes = list(data.get("resumes", []))
        self._started = datetime.fromisoformat(data["started"]) if data.get("started") else None
        self._md_path = json_path.parent / data["md_name"]

    def _resume_hint(self) -> str:
        assert self._md_path is not None
        return ("exe 를 더블클릭하면 이 토론을 이어서 할지 묻는다. 명령줄에서는:\n\n"
                f"    gemini_council.exe --resume \"{self._md_path}\"\n\n"
                "이어 하면 각 탭의 첫 메시지에 규칙 + 사회자 최신 요약 + 그 뒤 발언이 다시 들어가므로, "
                "같은 탭이든 새 채팅이든 상관없다.")

    SAVE_RETRIES = 3        # 파일이 다른 프로그램에 잠겨 있을 때 재시도 횟수
    SAVE_RETRY_S = 0.3      # 재시도 간격 [s]

    def _autosave(self) -> None:
        """발언마다 자동 저장. 파일이 잠겨 있으면(Windows 에서 다른 프로그램이 열어 둔 경우) 재시도하고,
        그래도 실패하면 경고만 하고 토론은 계속한다 (다음 발언 때 다시 저장). 최종 저장 실패는 숨기지 않는다."""
        for k in range(self.SAVE_RETRIES):
            try:
                self._save(STATUS_RUNNING, "진행 중 (발언마다 자동 저장)", self.clock())
                return
            except PermissionError as e:
                if k == self.SAVE_RETRIES - 1:
                    self.on_event(f"⚠ 자동 저장 실패 (파일이 다른 프로그램에서 열려 있음?): {e}. "
                                  "다음 발언 때 다시 저장합니다. 회의록 파일을 닫아 주세요.")
                    return
                time.sleep(self.SAVE_RETRY_S)

    def _save(self, status: str, reason_text: str, finished: datetime, detail: str = "") -> Path:
        assert self._md_path is not None and self._started is not None
        resumable = status not in ("COMPLETED", "MAX_ROUNDS")
        md = render_markdown(self.topic, self.state.utterances, self.cfg, reason_text, self._started, finished,
                             detail, appendix=self.backend.usage_report(), resumes=len(self._resumes),
                             resume_hint=self._resume_hint() if resumable else "",
                             abstract=self.abstract, abstract_note=self.abstract_note)
        write_atomic(self._md_path, md)
        save_state(state_path_for(self._md_path), {
            "version": STATE_VERSION, "status": status, "stop_reason": reason_text, "detail": detail,
            "topic": self.topic, "max_rounds": self.cfg.max_rounds, "char_limit": self.cfg.char_limit,
            "agenda": list(self.cfg.agenda), "roles": [r.key for r in self.roles],
            "started": self._started.isoformat(), "updated": finished.isoformat(),
            "resumes": self._resumes, "md_name": self._md_path.name,
            "utterances": utterances_to_state(self.state.utterances),
            "abstract": self.abstract,
        })
        return self._md_path

    def _write_abstract(self) -> None:
        """사회자에게 토론 전체 초록을 요청. 실패해도(q, 백엔드 오류) 토론 결과와 회의록은 그대로 둔다."""
        moderator = next(r for r in self.roles if r.is_moderator)
        prompt = build_abstract_prompt(self.roles, self.state, self.topic, self.cfg, self.backend.stateful)
        self.on_event(f"[초록] {moderator.name}에게 토론 전체 초록 요청 (입력 {len(prompt)}자)")
        self._autosave()  # 토론 본문은 초록 요청 전에 이미 저장
        try:
            self.abstract = self.backend.send(moderator, prompt).strip()
            self.on_event(f"[초록] 받음 ({len(self.abstract)}자)")
        except UserAbort:
            self.abstract_note = "초록 단계에서 사용자가 중단해 초록 없음 (토론 본문은 완결)"
            self.on_event("[초록] 건너뜀")
        except BackendError as e:
            self.abstract_note = f"초록 생성 실패: {e}"
            self.on_event(f"[초록] 실패: {e}")

    def _wait_if_paused(self) -> None:
        if not self.resume_event.is_set():
            self.on_event("일시정지됨")
            while not self.resume_event.wait(timeout=0.2):
                if self.stop_event.is_set():
                    return
            self.on_event("재개")

    def _loop(self) -> StopReason:
        n_roles = len(self.roles)
        done = len(self.state.utterances)  # 이어 하기면 이미 끝난 발언 수
        for r in range(done // n_roles + 1, self.cfg.max_rounds + 1):
            first_idx = done % n_roles if r == done // n_roles + 1 else 0
            for role in self.roles[first_idx:]:
                self._wait_if_paused()
                if self.stop_event.is_set():
                    return StopReason.USER_STOP
                recap = role.key in self._needs_recap
                prompt = build_turn_prompt(role, self.roles, self.state, r, self.topic,
                                           self.cfg, self.backend.stateful, recap=recap)
                tag = " (이어 하기: 맥락 복구 포함)" if recap else ""
                self.on_event(f"[R{r}] {role.name} 발언 요청 (입력 {len(prompt)}자){tag}")
                reply = self.backend.send(role, prompt)
                self._needs_recap.discard(role.key)
                utt = Utterance(r, role.key, role.name, reply)
                self.state.record(utt)
                self._autosave()
                warn = " ⚠ 길이 초과" if utt.chars > self.cfg.char_limit else ""
                self.on_event(f"[R{r}] {role.name} 응답 {utt.chars}자{warn}")
                if is_termination(role, reply, self.cfg):
                    return StopReason.COMPLETED
        return StopReason.MAX_ROUNDS

    def run(self) -> RunResult:
        now = self.clock()
        if self._resume is None:
            self._started = now
            # 시작하자마자 파일 이름을 확정 (이후 같은 파일을 계속 갱신, 다른 회의록은 덮어쓰지 않음)
            self._md_path = save_transcript("# 토론 회의록 (시작 중)\n", self.cfg.log_dir, now)
        else:
            self._resumes.append(now.isoformat())
            self.on_event(f"이어 하기: 발언 {len(self.state.utterances)}개 다음부터 진행 → {self._md_path}")
            if self._started is None:
                self._started = now
        self._save(STATUS_RUNNING, "진행 중 (발언마다 자동 저장)", now)
        reason, detail = StopReason.CRASH, ""
        crash: BaseException | None = None
        try:
            self.backend.open(self.roles)
            reason = self._loop()
            if reason in (StopReason.COMPLETED, StopReason.MAX_ROUNDS) and self.make_abstract:
                self._write_abstract()
        except UsageLimitReached as e:
            reason, detail = StopReason.USAGE_LIMIT, str(e)
        except UserAbort as e:
            reason, detail = StopReason.USER_STOP, str(e)
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
                log_path = self._save(reason.name, reason.value, self.clock(), detail)
                self.on_event(f"종료: {reason.value}. 회의록 저장: {log_path}")
                if reason.name not in ("COMPLETED", "MAX_ROUNDS"):
                    self.on_event("이어서 하려면 exe 를 다시 실행하면 이어 할지 묻습니다 "
                                  f"(명령줄: --resume \"{log_path}\")")
        if crash is not None:
            raise crash
        return RunResult(reason, self._complete_rounds(), tuple(self.state.utterances), log_path, detail)

    def _complete_rounds(self) -> int:
        """모든 역할이 발언을 마친 라운드 수 (중간에 끊긴 라운드는 제외)."""
        return len(self.state.utterances) // len(self.roles)
