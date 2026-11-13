"""PC 에 설치된 Chrome 을 전용 프로필 + 원격 디버깅 포트로 실행 (OS 분기는 이 파일 한 곳).

- 전용 프로필(--user-data-dir): 평소 쓰는 Chrome 프로필과 분리. 이 프로필에서 한 번 로그인하면 유지된다.
  (Chrome 136+ 는 기본 프로필에 원격 디버깅을 허용하지 않으므로 전용 프로필이 필수)
- 백그라운드 탭·가려진 창의 타이머/렌더링 감속을 끈다 (무인 실행 중 응답 완료 감지가 느려지지 않게).
- 자동화를 숨기는 옵션은 넣지 않는다.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_PORT = 9222
CHROME_FLAGS = (
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
)


class ChromeError(RuntimeError):
    """Chrome 을 찾거나 실행·연결할 수 없음."""


def default_profile_dir() -> Path:
    """전용 Chrome 프로필 폴더. Windows: %LOCALAPPDATA%\\gemini_council\\chrome_profile."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "gemini_council" / "chrome_profile"


def find_chrome() -> str:
    """Chrome 실행 파일 경로. 환경변수 COUNCIL_CHROME 이 있으면 그것을 쓴다."""
    env = os.environ.get("COUNCIL_CHROME")
    if env:
        if Path(env).exists():
            return env
        raise ChromeError(f"COUNCIL_CHROME 경로가 없음: {env}")
    candidates: list[Path] = []
    if sys.platform == "win32":
        for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(var)
            if root:
                candidates.append(Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe")
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    for c in candidates:
        if c.exists():
            return str(c)
    for name in ("google-chrome", "google-chrome-stable", "chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise ChromeError("Chrome 을 찾을 수 없음. Chrome 을 설치하거나 환경변수 COUNCIL_CHROME 에 chrome.exe 경로를 지정하라.")


def cdp_endpoint(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_cdp_alive(port: int, timeout_s: float = 1.0) -> bool:
    """디버깅 포트가 응답하는지 (이미 우리 프로필 Chrome 이 떠 있는지)."""
    try:
        with urllib.request.urlopen(cdp_endpoint(port) + "/json/version", timeout=timeout_s) as r:
            json.loads(r.read().decode("utf-8"))
        return True
    except (OSError, ValueError):
        return False


def launch_chrome(port: int = DEFAULT_PORT, profile_dir: Path | None = None, start_url: str = "about:blank",
                  extra_args: tuple[str, ...] = (), wait_s: float = 30.0) -> subprocess.Popen | None:
    """디버깅 포트가 이미 살아 있으면 그대로 쓰고(None 반환), 아니면 Chrome 을 띄우고 포트가 열릴 때까지 기다린다."""
    if is_cdp_alive(port):
        return None
    profile = profile_dir or default_profile_dir()
    profile.mkdir(parents=True, exist_ok=True)
    args = [find_chrome(), f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
            *CHROME_FLAGS, *extra_args, start_url]
    kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # 콘솔 Ctrl+C 가 Chrome 까지 죽이지 않게
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(args, **kw)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_cdp_alive(port, timeout_s=0.5):
            return proc
        if proc.poll() is not None:
            raise ChromeError(f"Chrome 이 바로 종료됨 (종료 코드 {proc.returncode}). "
                              "같은 프로필로 이미 열린 Chrome 창이 있으면 모두 닫고 다시 실행하라.")
        time.sleep(0.3)
    raise ChromeError(f"{wait_s:.0f}초 안에 Chrome 디버깅 포트({port})가 열리지 않음")
