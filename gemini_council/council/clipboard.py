"""OS 클립보드 읽기/쓰기 (OS 분기는 이 파일 한 곳). 외부 패키지 없음.

- Windows: ctypes 로 Win32 API 직접 호출. GetClipboardSequenceNumber 로 '복사가 일어났는지'를 감지한다
  (같은 내용을 다시 복사해도 번호가 바뀌므로 내용 비교보다 정확함).
- macOS: pbcopy / pbpaste.
- Linux: wl-copy/wl-paste (Wayland) 또는 xclip / xsel (X11) 중 설치된 것.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from typing import Protocol


class ClipboardError(RuntimeError):
    """클립보드를 쓸 수 없음 (다른 프로그램이 점유, 도구 미설치 등)."""


class Clipboard(Protocol):
    def get_text(self) -> str: ...
    def set_text(self, text: str) -> None: ...
    def change_token(self) -> object:
        """복사가 일어날 때마다 바뀌는 값. 지원하지 않으면 현재 내용 자체를 돌려준다."""
        ...


# ---------------------------------------------------------------- Windows

class WindowsClipboard:
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002
    OPEN_RETRIES = 20       # 다른 프로그램이 잠깐 점유 중일 때 재시도 횟수
    OPEN_RETRY_S = 0.05     # 재시도 간격 [s]

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        # 64비트에서 핸들/포인터가 잘리지 않도록 argtypes/restype 을 모두 명시한다
        u.OpenClipboard.argtypes, u.OpenClipboard.restype = [wintypes.HWND], wintypes.BOOL
        u.CloseClipboard.argtypes, u.CloseClipboard.restype = [], wintypes.BOOL
        u.EmptyClipboard.argtypes, u.EmptyClipboard.restype = [], wintypes.BOOL
        u.GetClipboardData.argtypes, u.GetClipboardData.restype = [wintypes.UINT], wintypes.HANDLE
        u.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        u.SetClipboardData.restype = wintypes.HANDLE
        u.GetClipboardSequenceNumber.argtypes, u.GetClipboardSequenceNumber.restype = [], wintypes.DWORD
        k.GlobalAlloc.argtypes, k.GlobalAlloc.restype = [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL
        k.GlobalLock.argtypes, k.GlobalLock.restype = [wintypes.HGLOBAL], wintypes.LPVOID
        k.GlobalUnlock.argtypes, k.GlobalUnlock.restype = [wintypes.HGLOBAL], wintypes.BOOL
        k.GlobalFree.argtypes, k.GlobalFree.restype = [wintypes.HGLOBAL], wintypes.HGLOBAL
        self._u, self._k = u, k

    def _open(self) -> None:
        for _ in range(self.OPEN_RETRIES):
            if self._u.OpenClipboard(None):
                return
            time.sleep(self.OPEN_RETRY_S)
        raise ClipboardError(f"클립보드를 열 수 없음 (다른 프로그램이 사용 중, 오류 {self._ct.get_last_error()})")

    def get_text(self) -> str:
        self._open()
        try:
            h = self._u.GetClipboardData(self.CF_UNICODETEXT)
            if not h:
                return ""  # 텍스트가 아닌 내용 (이미지 등)
            p = self._k.GlobalLock(h)
            if not p:
                raise ClipboardError("GlobalLock 실패")
            try:
                return self._ct.wstring_at(p)
            finally:
                self._k.GlobalUnlock(h)
        finally:
            self._u.CloseClipboard()

    def set_text(self, text: str) -> None:
        data = text.encode("utf-16-le") + b"\x00\x00"
        h = self._k.GlobalAlloc(self.GMEM_MOVEABLE, len(data))
        if not h:
            raise ClipboardError("GlobalAlloc 실패")
        p = self._k.GlobalLock(h)
        if not p:
            self._k.GlobalFree(h)
            raise ClipboardError("GlobalLock 실패")
        self._ct.memmove(p, data, len(data))
        self._k.GlobalUnlock(h)
        self._open()
        try:
            self._u.EmptyClipboard()
            if not self._u.SetClipboardData(self.CF_UNICODETEXT, h):
                self._k.GlobalFree(h)  # 실패 시에만 직접 해제 (성공하면 소유권이 시스템으로 넘어감)
                raise ClipboardError(f"SetClipboardData 실패 (오류 {self._ct.get_last_error()})")
        finally:
            self._u.CloseClipboard()

    def change_token(self) -> object:
        return int(self._u.GetClipboardSequenceNumber())


# ---------------------------------------------------------------- macOS / Linux (명령줄 도구)

class CommandClipboard:
    """pbcopy/xclip 등 명령줄 도구 기반. 변경 감지는 내용 비교로 한다."""

    def __init__(self, copy_cmd: list[str], paste_cmd: list[str]) -> None:
        self.copy_cmd, self.paste_cmd = copy_cmd, paste_cmd

    def get_text(self) -> str:
        r = subprocess.run(self.paste_cmd, capture_output=True, timeout=5)
        if r.returncode != 0:
            return ""  # 비어 있거나 텍스트가 아닌 내용
        return r.stdout.decode("utf-8", errors="replace")

    def set_text(self, text: str) -> None:
        # xclip/xsel/wl-copy 는 내용을 유지하려고 백그라운드 자식을 남긴다. 출력 파이프를 연결하면
        # 그 자식이 파이프를 쥐고 있어 run() 이 끝나지 않는다(실측: 5 s 타임아웃) → 출력은 버린다.
        r = subprocess.run(self.copy_cmd, input=text.encode("utf-8"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if r.returncode != 0:
            raise ClipboardError(f"{self.copy_cmd[0]} 실패 (종료 코드 {r.returncode})")

    def change_token(self) -> object:
        return self.get_text()


def system_clipboard() -> Clipboard:
    """현재 OS 의 클립보드 구현을 돌려준다."""
    if sys.platform == "win32":
        return WindowsClipboard()
    if sys.platform == "darwin":
        return CommandClipboard(["pbcopy"], ["pbpaste"])
    candidates = [
        (["wl-copy"], ["wl-paste", "--no-newline"]),
        (["xclip", "-selection", "clipboard"], ["xclip", "-selection", "clipboard", "-o"]),
        (["xsel", "--clipboard", "--input"], ["xsel", "--clipboard", "--output"]),
    ]
    for copy_cmd, paste_cmd in candidates:
        if shutil.which(copy_cmd[0]) and shutil.which(paste_cmd[0]):
            return CommandClipboard(copy_cmd, paste_cmd)
    raise ClipboardError("클립보드 도구가 없음. Ubuntu 는 `sudo apt install xclip` 후 다시 실행하라.")
