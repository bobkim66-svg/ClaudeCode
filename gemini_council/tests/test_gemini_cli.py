"""Gemini CLI 백엔드 테스트. 실제 CLI 대신 tests/fake_gemini.py 를 실행한다 (로그인 불필요)."""
import json
import sys
from pathlib import Path

import pytest

from council.backends import BackendError, ResponseTimeout, UsageLimitReached
from council.backends.gemini_cli import (
    GeminiCliBackend, GeminiCliConfig, classify_failure, extract_json_object, is_downgraded,
    summarize_stats,
)
from council.config import CouncilConfig
from council.orchestrator import Orchestrator, StopReason
from council.roles import DEFAULT_ROLES

FAKE = Path(__file__).with_name("fake_gemini.py")
P, C, A, M = DEFAULT_ROLES


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(log))
    monkeypatch.setenv("FAKE_GEMINI_MODE", "ok")
    monkeypatch.delenv("FAKE_GEMINI_END_AFTER", raising=False)
    return log


def _backend(timeout_s=30.0, **kw):
    cfg = GeminiCliConfig(command=(sys.executable, str(FAKE)), timeout_s=timeout_s, min_interval_s=0, **kw)
    return GeminiCliBackend(cfg)


def _calls(log):
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]


# ---------- 순수 파싱 함수 ----------

def test_extract_json_with_prefix_noise():
    text = 'Approval mode overridden...\n{\n "error": {"type": "X", "code": 41}\n}\ntrailer'
    assert extract_json_object(text)["error"]["code"] == 41
    assert extract_json_object("no json {broken") is None
    assert extract_json_object("") is None


def test_summarize_stats_and_downgrade():
    used, tot = summarize_stats({"models": {"gemini-3-pro-preview": {"tokens": {"total": 10}},
                                            "gemini-3.1-flash-lite": {"tokens": {"total": 5}}}})
    assert used == ("gemini-3-pro-preview", "gemini-3.1-flash-lite") and tot == 15
    assert summarize_stats(None) == ((), 0)
    assert summarize_stats({"models": "weird"}) == ((), 0)
    assert is_downgraded("pro", ("gemini-3-flash", "gemini-3.1-flash-lite"))
    assert not is_downgraded("pro", ("gemini-3-pro-preview", "gemini-3.1-flash-lite"))
    assert not is_downgraded("flash", ("gemini-3-flash",))
    assert not is_downgraded("pro", ())  # 정보 없으면 판단 보류


@pytest.mark.parametrize("rc,out,err,exc,needle", [
    (173, "", '{"error": {"type": "TerminalQuotaError", "message": "daily quota", "code": 429}}',
     UsageLimitReached, "한도"),
    (1, "", "Error: 429 RESOURCE_EXHAUSTED", UsageLimitReached, "한도"),
    (41, "", '{"error": {"type": "Error", "message": "Please set an Auth method", "code": 41}}',
     BackendError, "로그인"),
    (52, "", '{"error": {"type": "FatalConfigError", "message": "bad", "code": 52}}', BackendError, "52"),
    (1, "", "segfault", BackendError, "segfault"),
])
def test_classify_failure(rc, out, err, exc, needle):
    e = classify_failure(rc, out, err)
    assert type(e) is exc and needle in str(e)


def test_config_rejects_unsafe_model_name():
    with pytest.raises(ValueError):
        GeminiCliConfig(models={"fast": "flash & del *", "thinking": "pro"})
    with pytest.raises(ValueError):
        GeminiCliConfig(command=())


# ---------- 프로세스 호출 (가짜 CLI) ----------

def test_send_ok_passes_prompt_via_stdin_and_model_flag(fake_env):
    be = _backend()
    be.open(DEFAULT_ROLES)
    long_prompt = "가나다 " * 5000 + "\n줄바꿈 \"따옴표\" & | < > ^ %PATH%"  # 3만 자 + cmd 메타문자
    reply = be.send(C, long_prompt)
    call = _calls(fake_env)[0]
    assert call["stdin"] == long_prompt                       # 손실·이스케이프 없이 그대로 전달
    assert call["argv"][call["argv"].index("-m") + 1] == "pro"  # 비평가 = thinking → pro
    assert "-o" in call["argv"] and "json" in call["argv"]
    assert Path(call["cwd"]).name.startswith("council_ws_")     # 빈 임시 폴더에서 실행
    assert "한글 ✓" in reply
    assert be.calls[0].total_tokens > 0
    ws = Path(call["cwd"])
    be.close()
    assert not ws.exists()


def test_send_flash_for_fast_roles(fake_env):
    be = _backend()
    be.open(DEFAULT_ROLES)
    for r in (P, A, M):
        be.send(r, "x")
    assert [c["argv"][c["argv"].index("-m") + 1] for c in _calls(fake_env)] == ["flash"] * 3
    be.close()


def test_empty_model_omits_flag(fake_env):
    be = _backend(models={"fast": "", "thinking": ""})
    be.open(DEFAULT_ROLES)
    be.send(P, "x")
    assert "-m" not in _calls(fake_env)[0]["argv"]
    be.close()


@pytest.mark.parametrize("mode,exc", [
    ("quota", UsageLimitReached),
    ("quota_stdout", UsageLimitReached),
    ("auth", BackendError),
    ("crash", BackendError),
    ("garbage", BackendError),
    ("empty", BackendError),
])
def test_send_failures(fake_env, monkeypatch, mode, exc):
    monkeypatch.setenv("FAKE_GEMINI_MODE", mode)
    be = _backend()
    be.open(DEFAULT_ROLES)
    with pytest.raises(exc) as ei:
        be.send(P, "x")
    assert type(ei.value) is exc
    be.close()


def test_noise_before_json_is_tolerated(fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_MODE", "noise")
    be = _backend()
    be.open(DEFAULT_ROLES)
    assert "입장" in be.send(P, "x")
    be.close()


def test_timeout_kills_process(fake_env, monkeypatch):
    import time
    monkeypatch.setenv("FAKE_GEMINI_MODE", "hang")
    be = _backend(timeout_s=1.0)
    be.open(DEFAULT_ROLES)
    t0 = time.monotonic()
    with pytest.raises(ResponseTimeout):
        be.send(P, "x")
    assert time.monotonic() - t0 < 10  # 60s 기다리지 않고 바로 끊김
    be.close()


def test_downgrade_is_logged(fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_MODE", "downgrade")
    logs = []
    be = GeminiCliBackend(GeminiCliConfig(command=(sys.executable, str(FAKE)), min_interval_s=0), log=logs.append)
    be.open(DEFAULT_ROLES)
    be.send(C, "x")
    be.send(P, "x")
    assert sum("강등" in m for m in logs) == 1  # pro 요청한 비평가만 경고
    be.close()


def test_missing_command():
    be = GeminiCliBackend(GeminiCliConfig(command=("definitely-not-a-real-gemini-cmd",)))
    with pytest.raises(BackendError, match="찾을 수 없음"):
        be.open(DEFAULT_ROLES)


def test_send_before_open():
    with pytest.raises(BackendError):
        _backend().send(P, "x")


def test_pacing_waits_min_interval():
    """호출 간 최소 간격: 앞 호출 종료 0.5s 뒤에 다음 호출 → 1.5s 더 기다려야 함 (간격 2s)."""
    t = [100.0]
    slept = []
    be = GeminiCliBackend(GeminiCliConfig(min_interval_s=2.0), sleep=slept.append, clock=lambda: t[0])
    be._pace()
    assert slept == []          # 첫 호출은 대기 없음
    be._last_call_end = 100.0
    t[0] = 100.5
    be._pace()
    assert slept == [pytest.approx(1.5)]
    t[0] = 105.0
    be._pace()
    assert len(slept) == 1      # 이미 충분히 지남


# ---------- 오케스트레이터 통합 ----------

def test_full_discussion_with_fake_cli(fake_env, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_GEMINI_END_AFTER", "2")
    cfg = CouncilConfig(max_rounds=4, log_dir=tmp_path / "logs")
    res = Orchestrator(_backend(), "로봇 팔 역기구학 해법 비교", cfg).run()
    assert res.reason is StopReason.COMPLETED and res.rounds_done == 2
    calls = _calls(fake_env)
    assert len(calls) == 8
    assert all("[토론 규칙]" in c["stdin"] for c in calls)  # stateless → 매번 seed
    md = res.log_path.read_text(encoding="utf-8")
    assert "## 사용량" in md and "합계 (8회)" in md


def test_quota_mid_discussion_saves(fake_env, monkeypatch, tmp_path):
    cfg = CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")
    be = _backend()
    orch = Orchestrator(be, "주제", cfg)
    orig = be.send

    def send(role, prompt):
        if len(be.calls) == 5:
            monkeypatch.setenv("FAKE_GEMINI_MODE", "quota")
        return orig(role, prompt)

    be.send = send
    res = orch.run()
    assert res.reason is StopReason.USAGE_LIMIT and len(res.utterances) == 5
    assert "daily quota" in res.log_path.read_text(encoding="utf-8")
