"""OS 의존 동작 테스트: npm 식 래퍼 스크립트(gemini.cmd / sh) 경유 호출, 손자 프로세스 종료, 실행 파일 탐색.

npm 은 gemini 를 '래퍼 스크립트 → node' 2단 구조로 설치한다. 여기서는 래퍼 → python(fake_gemini) 로 흉내 낸다.
Windows 와 Linux 각각의 CI 에서 실제 OS 경로로 돈다.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from council.backends import ResponseTimeout
from council.backends.gemini_cli import GeminiCliBackend, GeminiCliConfig, resolve_executable
from council.roles import DEFAULT_ROLES

FAKE = Path(__file__).with_name("fake_gemini.py")
WIN = sys.platform == "win32"
P = DEFAULT_ROLES[0]


def _install_wrapper(bin_dir: Path) -> None:
    """bin_dir 에 npm 과 같은 구조의 gemini 래퍼를 만든다 (Windows 는 sh 파일 + .cmd 둘 다)."""
    bin_dir.mkdir(parents=True)
    sh = bin_dir / "gemini"
    # 마지막에 exit 를 둬서 셸이 python 을 exec 로 대체하지 않게 한다 → 진짜 '손자' 프로세스가 생김
    sh.write_text(f'#!/bin/sh\n"{sys.executable}" "{FAKE}" "$@"\nst=$?\nexit $st\n', encoding="utf-8", newline="\n")
    sh.chmod(0o755)
    if WIN:
        (bin_dir / "gemini.cmd").write_text(
            f'@echo off\r\n"{sys.executable}" "{FAKE}" %*\r\n', encoding="utf-8", newline="")


def _alive(pid: int) -> bool:
    if WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # 좀비(이미 종료, 부모가 회수 전)는 죽은 것으로 본다
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split()[2] == "Z")


@pytest.fixture
def wrapper_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "npm 전역 폴더"  # 공백 + 한글 경로
    _install_wrapper(bin_dir)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_GEMINI_MODE", "ok")
    return bin_dir


def test_wrapper_resolution(wrapper_on_path):
    exe = resolve_executable("gemini")
    assert exe is not None
    if WIN:
        assert exe.lower().endswith(".cmd")  # 확장자 없는 sh 파일이 아니라 .cmd 가 잡혀야 함


def test_call_through_wrapper_preserves_unicode(wrapper_on_path):
    be = GeminiCliBackend(GeminiCliConfig(min_interval_s=0, timeout_s=60))
    be.open(DEFAULT_ROLES)
    prompt = '한글 프롬프트 ✓ "따옴표" & | < > ^ %PATH% !x!\n두 번째 줄'
    reply = be.send(P, prompt)
    be.close()
    assert "한글 ✓" in reply
    import json
    log = json.loads((wrapper_on_path.parent / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert log["stdin"] == prompt


def test_timeout_kills_grandchild(wrapper_on_path, tmp_path, monkeypatch):
    pidfile = tmp_path / "pid.txt"
    monkeypatch.setenv("FAKE_GEMINI_MODE", "hang")
    monkeypatch.setenv("FAKE_GEMINI_PIDFILE", str(pidfile))
    be = GeminiCliBackend(GeminiCliConfig(min_interval_s=0, timeout_s=3))
    be.open(DEFAULT_ROLES)
    t0 = time.monotonic()
    with pytest.raises(ResponseTimeout):
        be.send(P, "x")
    assert time.monotonic() - t0 < 20
    be.close()
    pid = int(pidfile.read_text(encoding="utf-8"))
    for _ in range(50):  # 종료 반영까지 최대 5 s
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid), f"손자 프로세스(fake_gemini, pid={pid})가 살아 있음"


@pytest.mark.parametrize("found,expected", [
    ({"gemini.cmd": r"C:\npm\gemini.cmd", "gemini": r"C:\npm\gemini"}, r"C:\npm\gemini.cmd"),
    ({"gemini.exe": r"C:\x\gemini.exe"}, r"C:\x\gemini.exe"),
    ({"gemini": r"C:\npm\gemini"}, None),          # sh 스크립트만 있으면 실행 불가 → None
    ({"gemini.ps1": r"C:\npm\gemini.ps1"}, None),
    ({}, None),
])
def test_resolve_executable_windows_rules(found, expected):
    assert resolve_executable("gemini", which=found.get, windows=True) == expected


def test_resolve_executable_posix():
    assert resolve_executable("gemini", which={"gemini": "/usr/bin/gemini"}.get, windows=False) == "/usr/bin/gemini"
