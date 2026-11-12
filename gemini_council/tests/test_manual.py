"""반자동 중계 백엔드 테스트. 가짜 클립보드 + 대본대로 움직이는 가짜 사용자."""
import sys
from collections import deque

import pytest

from council.backends import UserAbort
from council.backends.manual import PROMPT_MARKERS, ManualConfig, ManualRelayBackend, check_reply
from council.config import CouncilConfig
from council.orchestrator import Orchestrator, StopReason
from council.protocol import DiscussionState, Utterance, build_turn_prompt
from council.roles import DEFAULT_ROLES

P, C, A, M = DEFAULT_ROLES


class FakeClipboard:
    """Windows 처럼 복사할 때마다 번호가 오르는 클립보드."""

    def __init__(self):
        self.text, self.seq, self.history = "", 0, []

    def get_text(self):
        return self.text

    def set_text(self, text):
        self.text, self.seq = text, self.seq + 1
        self.history.append(text)

    def change_token(self):
        return self.seq


class ScriptedUser:
    """매 폴링(sleep)마다 대본의 다음 동작을 수행. 동작: ("copy", 텍스트) | ("cmd", "r"/"q") | ("wait",)."""

    def __init__(self, clip, actions):
        self.clip, self.actions, self.cmds, self.t = clip, deque(actions), deque(), 0.0
        self.pasted = []  # 사용자가 각 차례에 붙여 넣은 내용 (= 그때의 클립보드)

    def sleep(self, dt):
        self.t += dt
        if not self.actions:
            raise AssertionError("대본이 끝났는데 백엔드가 계속 기다림")
        act = self.actions.popleft()
        if act[0] == "copy":
            self.clip.set_text(act[1])
        elif act[0] == "cmd":
            self.cmds.append(act[1])

    def command(self):
        return self.cmds.popleft() if self.cmds else None


def _backend(actions, cfg=ManualConfig()):
    clip = FakeClipboard()
    user = ScriptedUser(clip, actions)
    out = []
    be = ManualRelayBackend(clipboard=clip, out=out.append, commands=user.command, cfg=cfg,
                            sleep=user.sleep, clock=lambda: user.t)
    be.open(DEFAULT_ROLES)
    return be, clip, user, out


# ---------- 답변 판별 ----------

@pytest.mark.parametrize("text,ok", [
    ("(1) 입장: 반박\n(2) 근거: …", True),
    ("", False),
    ("   \n ", False),
    ("PROMPT", False),                                  # 보낸 메시지 그대로
    ("[현재] 라운드 1/3, 비평가의 차례입니다.", False),     # 보낸 메시지 일부를 복사
    ("이전 답변", False),                               # 직전 답변 재복사
])
def test_check_reply(text, ok):
    assert (check_reply(text, "PROMPT", "이전 답변") is None) is ok


def test_prompt_markers_match_protocol():
    """protocol.py 문구가 바뀌어 판별 표시가 어긋나면 여기서 잡힌다."""
    s = DiscussionState()
    first = build_turn_prompt(P, DEFAULT_ROLES, s, 1, "주제", CouncilConfig(), stateful=True)
    assert sum(m in first for m in PROMPT_MARKERS) >= 2
    for r in DEFAULT_ROLES:
        s.record(Utterance(1, r.key, r.name, "x"))
    later = build_turn_prompt(P, DEFAULT_ROLES, s, 2, "주제", CouncilConfig(), stateful=True)
    assert "[토론 규칙]" not in later and sum(m in later for m in PROMPT_MARKERS) >= 2


# ---------- 한 차례 ----------

def test_send_puts_prompt_on_clipboard_and_returns_copied_reply():
    be, clip, user, out = _backend([("wait",), ("wait",), ("copy", "  답변입니다 ✓  ")])
    reply = be.send(C, "보낼 프롬프트\n[현재] 라운드 1/3, 비평가의 차례입니다.")
    assert reply == "답변입니다 ✓"
    assert clip.history[0].startswith("보낼 프롬프트")       # 먼저 프롬프트를 클립보드에
    assert any("탭 2 · 비평가" in o for o in out)           # 비평가 = 2번 탭
    assert be.turns[0].tab == 2 and be.turns[0].reply_chars == len("답변입니다 ✓")


def test_wrong_copies_are_rejected_then_correct_one_accepted():
    prompt = "[현재] 라운드 1/3, 제안자의 차례입니다.\n본문"
    be, clip, user, out = _backend([
        ("copy", ""),                       # 이미지 등 비텍스트
        ("copy", prompt),                   # 프롬프트 재복사
        ("copy", "[현재] 라운드 1/3, 제안자의 차례입니다."),  # 자기 메시지 일부
        ("copy", "진짜 답변"),
    ])
    assert be.send(P, prompt) == "진짜 답변"
    assert sum("⚠" in o for o in out) == 3


def test_same_text_as_previous_reply_is_rejected():
    be, clip, user, out = _backend([("copy", "답 A"), ("copy", "답 A"), ("copy", "답 B")])
    assert be.send(P, "p1") == "답 A"
    assert be.send(C, "p2") == "답 B"
    assert any("직전 탭의 답변과 똑같음" in o for o in out)


def test_recopy_command():
    # 붙여 넣기 전에 이미지를 복사해 프롬프트를 잃음(비텍스트 → 답변으로 안 받음) → r 로 다시 복사
    be, clip, user, out = _backend([("copy", ""), ("cmd", "r"), ("wait",), ("copy", "답")])
    assert be.send(P, "보낼 프롬프트") == "답"
    assert clip.history.count("보낼 프롬프트") == 2  # 처음 + r 명령


def test_quit_command_raises_user_abort():
    be, *_ = _backend([("wait",), ("cmd", "q"), ("wait",)])
    with pytest.raises(UserAbort):
        be.send(P, "p")


def test_reminder_printed_while_waiting():
    cfg = ManualConfig(poll_s=1.0, reminder_s=5.0)
    be, clip, user, out = _backend([("wait",)] * 12 + [("copy", "답")], cfg=cfg)
    be.send(P, "p")
    assert sum("기다리는 중 (r:" in o for o in out) == 2  # 12 s 대기, 5 s 간격 → 2번


# ---------- 오케스트레이터 통합 ----------

class TabSimulator:
    """역할별 탭 4개를 흉내 내는 사용자: 클립보드의 프롬프트를 '붙여 넣고' 답변을 복사한다."""

    def __init__(self, clip, end_round=None):
        self.clip, self.end_round, self.t = clip, end_round, 0.0
        self.tab_inputs = {r.key: [] for r in DEFAULT_ROLES}
        self._pending = None

    def sleep(self, dt):
        self.t += dt
        prompt = self.clip.text
        role = next(r for r in DEFAULT_ROLES if f"{r.name}의 차례입니다." in prompt)
        self.tab_inputs[role.key].append(prompt)
        n = len(self.tab_inputs[role.key])
        reply = f"(1) 입장: 동의\n(2) 근거: {role.name} {n}라운드 ✓\n(3) 제안: 다음"
        if role.is_moderator and self.end_round and n >= self.end_round:
            reply += "\n\n[DISCUSSION_COMPLETE]"
        self.clip.set_text(reply)


def test_full_discussion_with_simulated_user(tmp_path):
    clip = FakeClipboard()
    sim = TabSimulator(clip, end_round=2)
    out = []
    be = ManualRelayBackend(clipboard=clip, out=out.append, sleep=sim.sleep, clock=lambda: sim.t)
    cfg = CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")
    res = Orchestrator(be, "주제", cfg).run()
    assert res.reason is StopReason.COMPLETED and res.rounds_done == 2 and len(res.utterances) == 8
    # stateful: 각 탭에 규칙은 첫 번째 붙여넣기에만, 두 번째부터는 새 발언만
    for key, inputs in sim.tab_inputs.items():
        assert "[토론 규칙]" in inputs[0] and "[토론 규칙]" not in inputs[1]
    # 2라운드 제안자 탭에는 1라운드 다른 3명의 발언이 모두 들어감
    p2 = sim.tab_inputs["proposer"][1]
    assert all(f"{n} 1라운드" in p2 for n in ("비평가", "대안 설계자", "사회자"))
    md = res.log_path.read_text(encoding="utf-8")
    assert "## 사용량" in md and "합계 (8회)" in md


def test_quit_mid_discussion_saves_transcript(tmp_path):
    clip = FakeClipboard()
    sim = TabSimulator(clip)
    cmds = deque()
    calls = {"n": 0}

    def sleep(dt):
        calls["n"] += 1
        if calls["n"] == 6:          # 6번째 차례에서 사용자가 q
            cmds.append("q")
            return
        sim.sleep(dt)

    be = ManualRelayBackend(clipboard=clip, out=lambda s: None,
                            commands=lambda: cmds.popleft() if cmds else None, sleep=sleep, clock=lambda: 0.0)
    res = Orchestrator(be, "주제", CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")).run()
    assert res.reason is StopReason.USER_STOP and len(res.utterances) == 5
    assert "q 로 중단" in res.log_path.read_text(encoding="utf-8")


# ---------- 실제 OS 클립보드 (Windows CI 에서만) ----------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows 클립보드 API 테스트")
def test_windows_clipboard_roundtrip():
    from council.clipboard import WindowsClipboard
    clip = WindowsClipboard()
    before = clip.get_text()
    sample = "한글 ✓ 😀 \"따옴표\"\r\n둘째 줄 " + "가" * 20000
    t0 = clip.change_token()
    clip.set_text(sample)
    try:
        assert clip.get_text() == sample
        assert clip.change_token() != t0
    finally:
        clip.set_text(before)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX 명령줄 클립보드 도구 경로")
def test_command_clipboard_set_does_not_hang_on_daemonizing_tool(tmp_path):
    """xclip 처럼 백그라운드 자식을 남기는 도구에도 set_text 가 바로 끝나야 함 (실측 회귀: 5 s 타임아웃)."""
    import time
    from council.clipboard import CommandClipboard
    store = tmp_path / "clip.txt"
    tool = tmp_path / "fakeclip"
    tool.write_text(f'#!/bin/sh\ncat > "{store}"\nsleep 30 &\nexit 0\n', encoding="utf-8")  # 자식이 stdout 상속
    tool.chmod(0o755)
    clip = CommandClipboard([str(tool)], ["cat", str(store)])
    t0 = time.monotonic()
    clip.set_text("한글 ✓")
    assert time.monotonic() - t0 < 2.0
    assert clip.get_text() == "한글 ✓"


# ---------- 차례 사이 최소 간격 ----------

class ClockUser:
    """가짜 시계 + 대본. sleep 할 때마다 시간이 흐르고, 지정 시각에 동작을 수행한다."""

    def __init__(self, clip, events):
        self.clip, self.t, self.events, self.cmds = clip, 0.0, sorted(events, key=lambda e: e[0]), deque()
        self.prompt_times = []
        self._seen_seq = clip.seq

    def sleep(self, dt):
        self.t += dt
        while self.events and self.events[0][0] <= self.t:
            _, kind, val = self.events.pop(0)
            if kind == "copy":
                self.clip.set_text(val)
            else:
                self.cmds.append(val)

    def command(self):
        return self.cmds.popleft() if self.cmds else None


def _interval_backend(events, interval_s=300.0):
    clip = FakeClipboard()
    user = ClockUser(clip, events)
    out = []
    be = ManualRelayBackend(clipboard=clip, out=out.append, commands=user.command,
                            cfg=ManualConfig(poll_s=1.0, turn_interval_s=interval_s, countdown_s=60.0),
                            sleep=user.sleep, clock=lambda: user.t)
    be.open(DEFAULT_ROLES)
    return be, clip, user, out


def test_turn_interval_enforced_and_clipboard_untouched_while_waiting():
    # 1번째 프롬프트 t=0, 답변 t=30 → 2번째 프롬프트는 t=300 이후에야 클립보드에
    be, clip, user, out = _interval_backend([(30, "copy", "답 1"), (100, "copy", "기다리는 중 다른 복사"),
                                             (330, "copy", "답 2")])
    assert be.send(P, "프롬프트 1") == "답 1"
    issued_at = []
    orig_set = clip.set_text

    def spy(text):
        if text.startswith("프롬프트"):  # 프로그램이 넣는 것만 기록 (가짜 사용자의 복사는 제외)
            issued_at.append((user.t, text))
        orig_set(text)
    clip.set_text = spy
    assert be.send(C, "프롬프트 2") == "답 2"
    assert issued_at[0][1] == "프롬프트 2" and issued_at[0][0] >= 300.0      # 5분 전에는 클립보드에 안 넣음
    assert any("다음 차례 [탭 2 · 비평가] 까지 4분 30초 대기" in o for o in out)
    assert sum("남음" in o for o in out) == 4                                  # 1분마다 남은 시간 (4:00, 3:00, 2:00, 1:00 시점)


def test_first_turn_does_not_wait_and_zero_interval_disables():
    be, clip, user, out = _interval_backend([(5, "copy", "답")])
    be.send(P, "p")
    assert not any("대기" in o and "⏸" in o for o in out)
    be0, clip0, user0, out0 = _interval_backend([(5, "copy", "답 A"), (10, "copy", "답 B")], interval_s=0)
    be0.send(P, "p1")
    be0.send(C, "p2")
    assert not any("⏸" in o for o in out0) and user0.t < 15


def test_skip_wait_with_n_and_quit_with_q():
    be, clip, user, out = _interval_backend([(10, "copy", "답 1"), (40, "cmd", "n"), (45, "copy", "답 2")])
    be.send(P, "p1")
    assert be.send(C, "p2") == "답 2" and user.t < 60
    assert any("대기를 건너뜁니다" in o for o in out)
    be2, *_ = _interval_backend([(10, "copy", "답 1"), (40, "cmd", "q")])
    be2.send(P, "p1")
    with pytest.raises(UserAbort):
        be2.send(C, "p2")


def test_interval_validation():
    with pytest.raises(ValueError):
        ManualConfig(turn_interval_s=-1)


def test_full_discussion_interval_total_time(tmp_path):
    """12차례(3라운드)면 간격 대기는 11번 → 마지막 프롬프트는 최소 11×300 s 뒤에 나간다."""
    clip = FakeClipboard()
    sim = TabSimulator(clip)
    issue_times = []
    orig = clip.set_text

    def set_text(text):
        if "의 차례입니다." in text:
            issue_times.append(sim.t)
        orig(text)
    clip.set_text = set_text

    def sleep(dt):
        sim.t += dt
        if clip.text and "의 차례입니다." in clip.text and sim.t >= issue_times[-1] + 20:  # 20 s 뒤 답변
            sim.sleep(0)

    be = ManualRelayBackend(clipboard=clip, out=lambda s: None, sleep=sleep, clock=lambda: sim.t,
                            cfg=ManualConfig(poll_s=1.0, turn_interval_s=300.0))
    res = Orchestrator(be, "주제", CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")).run()
    assert len(res.utterances) == 12 and len(issue_times) == 12
    gaps = [b - a for a, b in zip(issue_times, issue_times[1:])]
    assert min(gaps) >= 300.0 - 1e-9 and issue_times[-1] >= 11 * 300.0
