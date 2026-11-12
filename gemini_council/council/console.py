"""콘솔 줄 입력을 백그라운드 스레드로 받아, 클립보드 감시와 동시에 'r'/'q' 같은 명령을 처리한다."""
from __future__ import annotations

import queue
import sys
import threading
from typing import TextIO


class ConsoleCommands:
    """stdin 한 줄 = 명령 하나. poll() 은 기다리지 않고, wait_line() 은 한 줄이 올 때까지 기다린다.

    스레드는 데몬이라 프로그램 종료를 막지 않는다. 같은 stdin 을 input() 과 동시에 읽으면
    줄을 서로 빼앗으므로, 이 객체를 만든 뒤에는 모든 입력을 이 객체로 받는다.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self._stream = stream or sys.stdin
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self) -> None:
        for line in self._stream:
            self._q.put(line.rstrip("\r\n"))
        self._q.put(None)  # EOF

    def poll(self) -> str | None:
        """대기 중인 명령 한 줄 (소문자, 앞뒤 공백 제거). 없으면 None."""
        try:
            line = self._q.get_nowait()
        except queue.Empty:
            return None
        return None if line is None else line.strip().lower()

    def wait_line(self, timeout: float | None = None) -> str | None:
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return line
