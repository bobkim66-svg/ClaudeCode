"""Google 공식 Gemini CLI 를 헤드리스(-p)로 호출하는 백엔드.

확인한 CLI 동작 (@google/gemini-cli 0.68.0 소스와 실제 실행으로 확인):
- `-p TEXT`: 비대화형 모드. TEXT 는 stdin 입력 뒤에 덧붙는다 → 긴 프롬프트는 stdin 으로 보낸다.
  (명령줄 길이 제한, Windows .cmd 인자 이스케이프 문제를 피하기 위함)
- `-o json`: 성공 시 stdout 에 {"session_id", "response", "stats": {"models": {...}}}.
  실패 시 stderr 에 {"error": {"type", "message", "code"}} (앞에 안내 문구가 섞일 수 있음).
- 종료 코드: 41 인증 없음, 42 입력 오류, 52 설정 오류, 53 턴 제한, 55 신뢰되지 않은 폴더.
  한도 초과는 HTTP 상태(429)를 종료 코드로 쓰므로 POSIX 에서 173(=429 mod 256)으로 잘린다
  → 종료 코드가 아니라 error.type ("TerminalQuotaError"/"RetryableQuotaError")로 판별한다.
- 헤드리스 + 기본 승인 모드에서는 '승인 필요' 도구(셸 실행, 파일 쓰기/수정)가 자동 거부된다.
  추가로 빈 임시 폴더에서 실행해 읽기 도구가 볼 파일도 없게 한다. --yolo 는 절대 쓰지 않는다.
- `-m pro` / `-m flash` 별칭은 계정 권한에 맞는 실제 모델로 CLI 가 해석한다.

매 호출이 독립적이므로 stateless 백엔드다 (맥락은 protocol.py 가 프롬프트에 담는다).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..roles import Role
from .base import BackendError, ChatBackend, ResponseTimeout, UsageLimitReached

EXIT_AUTH = 41
QUOTA_ERROR_TYPES = {"TerminalQuotaError", "RetryableQuotaError"}
QUOTA_PATTERN = re.compile(r"quota|RESOURCE_EXHAUSTED|rate limit|\b429\b", re.IGNORECASE)
# 모델 이름은 명령줄 인자로 들어가므로 안전한 문자만 허용 (Windows cmd 메타문자 주입 방지)
SAFE_MODEL = re.compile(r"^[A-Za-z0-9._-]+$")
HEADLESS_FLAG_TEXT = "위 지시에 따라 지금 발언하라."


@dataclass(frozen=True)
class GeminiCliConfig:
    """Gemini CLI 백엔드 설정.

    Attributes:
        command: 실행 명령. 첫 요소는 PATH 에서 찾는다. 예: ("gemini",), ("npx", "@google/gemini-cli").
        models: Role.model_hint → -m 값. 별칭("flash", "pro") 권장. 빈 문자열이면 -m 생략(CLI 기본).
        timeout_s: 호출 1회 제한 시간 [s]. CLI 내부 재시도(한도 백오프) 시간도 포함된다.
        min_interval_s: 연속 호출 사이 최소 간격 [s]. 분당 요청 한도에 몰아서 부딪히지 않게 한다.
    """

    command: tuple[str, ...] = ("gemini",)
    models: dict[str, str] = field(default_factory=lambda: {"fast": "flash", "thinking": "pro"})
    timeout_s: float = 180.0
    min_interval_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command 가 비어 있음")
        if self.timeout_s <= 0 or self.min_interval_s < 0:
            raise ValueError("timeout_s > 0, min_interval_s >= 0 이어야 함")
        for hint, m in self.models.items():
            if m and not SAFE_MODEL.match(m):
                raise ValueError(f"모델 이름에 허용되지 않는 문자: {hint}={m!r}")


@dataclass(frozen=True)
class CallRecord:
    """호출 1회 기록 (회의록 사용량 표에 쓰임). seconds 단위 [s]."""

    role_name: str
    requested_model: str
    used_models: tuple[str, ...]
    total_tokens: int
    seconds: float


# ---------------------------------------------------------------- 출력 파싱 (순수 함수)

def extract_json_object(text: str) -> dict | None:
    """text 안에서 처음으로 파싱되는 JSON 객체를 찾는다 (앞뒤 안내 문구 허용). 없으면 None."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def summarize_stats(stats: dict | None) -> tuple[tuple[str, ...], int]:
    """stats.models 에서 (사용된 모델 이름들, 총 토큰 수)를 뽑는다. 구조가 다르면 ((), 0)."""
    models = (stats or {}).get("models") or {}
    if not isinstance(models, dict):
        return (), 0
    total = 0
    for v in models.values():
        tok = (v or {}).get("tokens", {}) if isinstance(v, dict) else {}
        total += int(tok.get("total", 0) or 0)
    return tuple(sorted(models)), total


def is_downgraded(requested: str, used: tuple[str, ...]) -> bool:
    """'pro' 를 요청했는데 이름에 'pro' 가 들어간 모델이 하나도 쓰이지 않았으면 강등으로 본다.

    CLI 는 내부 보조 호출(라우팅 등)에 flash 계열을 함께 쓸 수 있으므로 '포함 여부'로만 판단한다.
    """
    if not used or "pro" not in requested.lower():
        return False
    return not any("pro" in u.lower() for u in used)


def classify_failure(returncode: int, stdout: str, stderr: str) -> BackendError:
    """실패한 호출의 출력에서 적절한 예외를 만든다."""
    obj = extract_json_object(stderr) or extract_json_object(stdout) or {}
    err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
    etype = str(err.get("type", ""))
    msg = str(err.get("message", "")) or (stderr.strip() or stdout.strip())[-800:]
    if etype in QUOTA_ERROR_TYPES or QUOTA_PATTERN.search(msg):
        return UsageLimitReached(f"Gemini 사용량 한도: {msg[:500]}")
    if returncode == EXIT_AUTH or err.get("code") == EXIT_AUTH:
        return BackendError(
            "Gemini CLI 로그인이 필요함. 터미널에서 `gemini` 를 한 번 대화형으로 실행해 "
            f"'Login with Google' 로 로그인한 뒤 다시 실행하라. (원문: {msg[:300]})")
    return BackendError(f"Gemini CLI 실패 (종료 코드 {returncode}, {etype or '형식 불명'}): {msg[:500]}")


# ---------------------------------------------------------------- 프로세스 실행 (OS 분기는 여기 한 곳)

WINDOWS_RUNNABLE = (".exe", ".cmd", ".bat", ".com")


def resolve_executable(name: str, which: Callable[[str], str | None] = shutil.which,
                       windows: bool = sys.platform == "win32") -> str | None:
    """실행 파일 경로를 찾는다.

    Windows 에서 npm 은 같은 폴더에 gemini (확장자 없는 sh 스크립트), gemini.cmd, gemini.ps1 을 만든다.
    확장자 없는 파일이 잡히면 CreateProcess 가 실패(WinError 193)하므로 .cmd/.exe 를 우선한다.
    """
    if windows and Path(name).suffix.lower() not in WINDOWS_RUNNABLE:
        for ext in WINDOWS_RUNNABLE:
            found = which(name + ext)
            if found:
                return found
    found = which(name)
    if windows and found and Path(found).suffix.lower() not in WINDOWS_RUNNABLE:
        return None  # 실행할 수 없는 파일 (sh 스크립트, .ps1 등)
    return found


def _kill_tree(proc: subprocess.Popen) -> None:
    """자식 프로세스 트리 전체 종료. Windows 의 gemini.cmd → node 처럼 손자 프로세스가 남는 것을 막는다."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def run_process(argv: list[str], stdin_text: str, timeout_s: float, cwd: Path) -> tuple[int, str, str]:
    """argv 실행, stdin 에 UTF-8 텍스트 입력. (종료 코드, stdout, stderr) 반환.

    Raises:
        subprocess.TimeoutExpired: 제한 시간 초과 (프로세스 트리는 종료된 상태).
    """
    kw: dict = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True  # 새 프로세스 그룹 → killpg 로 한 번에 종료
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=str(cwd), **kw)
    try:
        out, err = proc.communicate(stdin_text.encode("utf-8"), timeout=timeout_s)
    except BaseException:  # 타임아웃·Ctrl+C 모두: 자식 정리 후 그대로 다시 던짐
        _kill_tree(proc)
        raise
    return proc.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- 백엔드

class GeminiCliBackend(ChatBackend):
    """Gemini CLI 헤드리스 호출 백엔드 (stateless)."""

    stateful = False

    def __init__(
        self,
        cfg: GeminiCliConfig | None = None,
        log: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg or GeminiCliConfig()
        self.log = log or (lambda msg: None)
        self._sleep, self._clock = sleep, clock
        self._argv0: list[str] = []
        self._workdir: Path | None = None
        self._last_call_end: float | None = None
        self.calls: list[CallRecord] = []

    def _resolve_command(self) -> list[str]:
        exe = resolve_executable(self.cfg.command[0])
        if exe is None:
            raise BackendError(
                f"'{self.cfg.command[0]}' 명령을 찾을 수 없음. Node.js 20+ 설치 후 "
                "`npm install -g @google/gemini-cli` 로 설치하거나 --gemini-cmd 로 경로를 지정하라.")
        return [exe, *self.cfg.command[1:]]

    def open(self, roles: tuple[Role, ...]) -> None:
        for r in roles:
            if r.model_hint not in self.cfg.models:
                raise BackendError(f"역할 '{r.name}' 의 model_hint '{r.model_hint}' 에 대응하는 모델 설정 없음")
        self._argv0 = self._resolve_command()
        # 빈 임시 작업 폴더: 프로젝트 파일/GEMINI.md 가 맥락에 섞이지 않고, 읽기 도구가 볼 파일도 없다.
        self._workdir = Path(tempfile.mkdtemp(prefix="council_ws_"))
        self.log(f"Gemini CLI: {' '.join(self._argv0)} (작업 폴더 {self._workdir})")

    def close(self) -> None:
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)  # 임시 폴더 정리 실패는 결과에 영향 없음
            self._workdir = None

    def build_argv(self, model: str) -> list[str]:
        argv = [*self._argv0, "-p", HEADLESS_FLAG_TEXT, "-o", "json"]
        if model:
            argv += ["-m", model]
        return argv

    def _pace(self) -> None:
        if self._last_call_end is None:
            return
        wait = self.cfg.min_interval_s - (self._clock() - self._last_call_end)
        if wait > 0:
            self._sleep(wait)

    def send(self, role: Role, prompt: str) -> str:
        if self._workdir is None:
            raise BackendError("open() 전에 send() 호출됨")
        model = self.cfg.models[role.model_hint]
        self._pace()
        t0 = self._clock()
        try:
            rc, out, err = run_process(self.build_argv(model), prompt, self.cfg.timeout_s, self._workdir)
        except subprocess.TimeoutExpired as e:
            raise ResponseTimeout(f"{role.name}: {self.cfg.timeout_s:.0f}s 안에 응답 없음") from e
        finally:
            self._last_call_end = self._clock()
        elapsed = self._last_call_end - t0

        if rc != 0:
            raise classify_failure(rc, out, err)
        obj = extract_json_object(out)
        if obj is None:
            raise BackendError(f"Gemini CLI 출력에서 JSON 을 찾지 못함: {out[:300]!r}")
        if "error" in obj:
            raise classify_failure(rc, out, err)
        reply = obj.get("response")
        if not isinstance(reply, str) or not reply.strip():
            raise BackendError(f"{role.name}: 빈 응답을 받음")

        used, tokens = summarize_stats(obj.get("stats"))
        self.calls.append(CallRecord(role.name, model or "(기본)", used, tokens, elapsed))
        if is_downgraded(model, used):
            self.log(f"⚠ {role.name}: '{model}' 요청했으나 사용된 모델 {used} (한도로 인한 강등 가능성)")
        return reply.strip()

    def usage_report(self) -> str:
        if not self.calls:
            return ""
        rows = ["| 발언자 | 요청 모델 | 사용된 모델 | 토큰 | 시간 [s] |", "|---|---|---|---:|---:|"]
        for c in self.calls:
            rows.append(f"| {c.role_name} | {c.requested_model} | {', '.join(c.used_models) or '-'} "
                        f"| {c.total_tokens} | {c.seconds:.1f} |")
        total = sum(c.total_tokens for c in self.calls)
        rows.append(f"| 합계 ({len(self.calls)}회) | | | {total} | {sum(c.seconds for c in self.calls):.1f} |")
        return "\n".join(rows)
