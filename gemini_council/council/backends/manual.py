"""반자동 중계 백엔드: 사람이 Gemini 웹 탭과 프로그램 사이에서 복사·붙여넣기만 한다.

Gemini 화면은 프로그램이 전혀 건드리지 않는다 (자동 조작 없음). 프로그램은 로컬 클립보드만 다룬다:
  1) 보낼 프롬프트를 클립보드에 넣고 '몇 번 탭에 붙여 넣으라'고 안내
  2) 사용자가 Gemini 답변의 '복사' 버튼을 누르면 클립보드 변경을 감지해 답변으로 받는다
웹 탭이 대화를 기억하므로 stateful 백엔드다 (각 탭에 처음엔 규칙+역할, 이후엔 새 발언만 보냄).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..clipboard import Clipboard, system_clipboard
from ..roles import Role
from .base import ChatBackend, UserAbort

# 프로그램이 만든 프롬프트에만 있는 표시. 답변 대신 보낸 메시지를 복사했는지 판별에 쓴다.
# (protocol.py 의 문구와 어긋나면 tests/test_manual.py 가 잡는다)
PROMPT_MARKERS = ("[토론 규칙]", "[지난 발언 이후 새 발언]", "의 차례입니다.", "[새 토론 시작", "[토론 종료 — 초록 작성]")

# 같은 탭을 여러 토론에 재활용하므로, 이번 실행에서 각 탭의 첫 메시지 앞에 붙여 이전 대화를 무시하게 한다.
# (지시에 의한 리셋일 뿐 이전 내용은 Gemini 맥락에 남아 있다 → 완전한 리셋은 새 채팅)
RESET_PREFIX = ("[새 토론 시작 — 이전 대화 무시]\n"
                "이 채팅의 위쪽 내용은 이전 토론입니다. 이전 주제·결론·역할 지시를 모두 잊고, "
                "이 메시지의 주제와 규칙만 따르세요. 이전 토론을 언급하거나 이어 쓰지 마세요.\n\n")


@dataclass(frozen=True)
class ManualConfig:
    """poll_s: 클립보드 확인 간격 [s]. reminder_s: 안내 재출력 간격 [s]. preview_chars: 미리보기 길이 [자].

    turn_interval_s: 연속한 두 차례의 메시지를 내보내는 최소 간격 [s] (이전 메시지를 클립보드에 넣은 시점 기준).
        Gemini 에 메시지가 몰려 나가지 않게 한다. 0 이면 대기 없음.
    countdown_s: 대기 중 남은 시간 표시 간격 [s].
    """

    poll_s: float = 0.3
    reminder_s: float = 120.0
    preview_chars: int = 60
    turn_interval_s: float = 0.0
    countdown_s: float = 60.0

    def __post_init__(self) -> None:
        if self.turn_interval_s < 0:
            raise ValueError(f"turn_interval_s 는 0 이상이어야 함: {self.turn_interval_s}")


@dataclass(frozen=True)
class TurnRecord:
    role_name: str
    tab: int
    sent_chars: int
    reply_chars: int
    wait_s: float


def _preview(text: str, n: int) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n] + "…"


def _fmt_mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}분 {s % 60:02d}초"


def check_reply(text: str, prompt: str, previous: str | None) -> str | None:
    """복사된 내용이 이번 답변으로 적절한지 검사. 적절하면 None, 아니면 사용자에게 보일 이유."""
    s = text.strip()
    if not s:
        return "클립보드가 비어 있거나 텍스트가 아님"
    if s == prompt.strip():
        return "보낼 메시지가 그대로 복사됨 (아직 붙여 넣기 전이면 정상)"
    if any(m in s for m in PROMPT_MARKERS):
        return "내가 보낸 메시지를 복사한 것 같음. 답변 맨 아래 아이콘 줄(좋아요·싫어요·다시 생성·복사·더보기) 중 4번째 '겹친 네모' 복사 아이콘을 눌러야 함"
    if previous is not None and s == previous.strip():
        return "직전 탭의 답변과 똑같음. 이번 탭의 새 답변을 복사했는지 확인"
    return None


class ManualRelayBackend(ChatBackend):
    """사람이 중계하는 백엔드. 시간 제한 없이 기다린다 ('q' 로 저장 후 종료)."""

    stateful = True

    def __init__(
        self,
        clipboard: Clipboard | None = None,
        out: Callable[[str], None] = print,
        commands: Callable[[], str | None] = lambda: None,
        cfg: ManualConfig = ManualConfig(),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clip = clipboard
        self.out, self.commands, self.cfg = out, commands, cfg
        self._sleep, self._clock = sleep, clock
        self._tabs: dict[str, int] = {}
        self._last_reply: str | None = None
        self._last_issue: float | None = None  # 직전 메시지를 클립보드에 넣은 시각 [s]
        self._reset_done: set[str] = set()       # 이번 실행에서 리셋 문구를 이미 보낸 탭
        self.turns: list[TurnRecord] = []

    def open(self, roles: tuple[Role, ...]) -> None:
        if self._clip is None:
            self._clip = system_clipboard()
        self._tabs = {r.key: i + 1 for i, r in enumerate(roles)}
        lines = ["", "=" * 64, " 반자동 모드 준비", "=" * 64,
                 f" Chrome 에서 Gemini 새 채팅 탭을 {len(roles)}개 열고 아래 순서로 정하세요:"]
        for r in roles:
            hint = " (Pro/사고 모드 권장)" if r.model_hint == "thinking" else ""
            lines.append(f"   탭 {self._tabs[r.key]}: {r.name}{hint}")
        lines += [" 이전 토론에 쓰던 탭을 그대로 써도 됩니다 (각 탭 첫 메시지에 '새 토론 시작' 리셋 문구가 붙음).",
                  " 진행 방법 (매 차례 반복):",
                  "   1) 안내된 탭에서 Ctrl+V, Enter",
                  "   2) 답변이 끝나면 답변 맨 아래 아이콘 줄(좋아요·싫어요·다시 생성·복사·더보기) 중 4번째 '겹친 네모' 복사 아이콘 클릭",
                  "      → 프로그램이 자동으로 다음 차례로 넘어감 (코드 블록 위의 복사 버튼, 내 질문의 복사 버튼 아님)",
                  " ※ 기다리는 동안 다른 글을 복사하면 그 글이 답변으로 처리됩니다.",
                  " 명령: r + Enter = 보낼 메시지 다시 복사,  q + Enter = 저장 후 종료",
                  "=" * 64, ""]
        self.out("\n".join(lines))

    def send(self, role: Role, prompt: str) -> str:
        assert self._clip is not None, "open() 전에 send() 호출됨"
        tab = self._tabs.get(role.key, 0)
        self._wait_turn_interval(tab, role.name)
        if role.key not in self._reset_done:
            prompt = RESET_PREFIX + prompt
            self._reset_done.add(role.key)
        self._clip.set_text(prompt)
        self._last_issue = self._clock()
        baseline = self._clip.change_token()
        self.out(f"\n▶ [탭 {tab} · {role.name}] 에 붙여 넣으세요 (Ctrl+V, Enter) — {len(prompt)}자 복사됨")
        header = next((ln for ln in prompt.splitlines() if "의 차례입니다." in ln), prompt)
        self.out(f"   보낼 내용: {_preview(header, self.cfg.preview_chars)}")
        self.out("   답변이 끝나면 답변 맨 아래 아이콘 줄(좋아요·싫어요·다시 생성·복사·더보기) 중 4번째 '겹친 네모' 복사 아이콘을 누르세요. 기다리는 중…")

        t0 = last_note = self._clock()
        while True:
            cmd = self.commands()
            if cmd == "q":
                raise UserAbort("사용자가 q 로 중단")
            if cmd == "r":
                self._clip.set_text(prompt)
                baseline = self._clip.change_token()
                self.out(f"   ↻ 보낼 메시지를 다시 복사했습니다 ({len(prompt)}자). [탭 {tab}] 에 붙여 넣으세요.")
            tok = self._clip.change_token()
            if tok != baseline:
                baseline = tok
                text = self._clip.get_text()
                problem = check_reply(text, prompt, self._last_reply)
                if problem is None:
                    reply = text.strip()
                    wait = self._clock() - t0
                    self._last_reply = reply
                    self.turns.append(TurnRecord(role.name, tab, len(prompt), len(reply), wait))
                    self.out(f"   ✔ 답변 받음 ({len(reply)}자, {wait:.0f}초): {_preview(reply, self.cfg.preview_chars)}")
                    return reply
                self.out(f"   ⚠ {problem}")
            if self._clock() - last_note >= self.cfg.reminder_s:
                last_note = self._clock()
                self.out(f"   … [탭 {tab} · {role.name}] 답변 복사를 기다리는 중 (r: 다시 복사, q: 저장 후 종료)")
            self._sleep(self.cfg.poll_s)

    def _wait_turn_interval(self, tab: int, role_name: str) -> None:
        """직전 메시지를 내보낸 뒤 turn_interval_s 가 지날 때까지 기다린다.

        기다리는 동안에는 다음 메시지를 클립보드에 넣지 않는다 (미리 붙여 넣는 실수 방지).
        n + Enter = 대기 건너뛰기, q + Enter = 저장 후 종료.
        """
        if self._last_issue is None or self.cfg.turn_interval_s <= 0:
            return
        deadline = self._last_issue + self.cfg.turn_interval_s
        remaining = deadline - self._clock()
        if remaining <= 0:
            return
        self.out(f"\n⏸ 다음 차례 [탭 {tab} · {role_name}] 까지 {_fmt_mmss(remaining)} 대기 "
                 "(Gemini 창은 그대로 두세요. n + Enter: 바로 진행, q + Enter: 저장 후 종료)")
        last_note = self._clock()
        while True:
            cmd = self.commands()
            if cmd == "q":
                raise UserAbort("사용자가 q 로 중단")
            if cmd == "n":
                self.out("   대기를 건너뜁니다.")
                return
            now = self._clock()
            if now >= deadline:
                return
            if now - last_note >= self.cfg.countdown_s:
                last_note = now
                self.out(f"   … {_fmt_mmss(deadline - now)} 남음")
            self._sleep(min(self.cfg.poll_s, max(deadline - now, 0.0)))

    def usage_report(self) -> str:
        if not self.turns:
            return ""
        rows = ["| 탭 | 발언자 | 보낸 글자 | 받은 글자 | 대기 [s] |", "|---:|---|---:|---:|---:|"]
        rows += [f"| {t.tab} | {t.role_name} | {t.sent_chars} | {t.reply_chars} | {t.wait_s:.0f} |" for t in self.turns]
        rows.append(f"| | 합계 ({len(self.turns)}회) | {sum(t.sent_chars for t in self.turns)} "
                    f"| {sum(t.reply_chars for t in self.turns)} | {sum(t.wait_s for t in self.turns):.0f} |")
        return "\n".join(rows)
