"""자동(웹) 모드 테스트: 실제 Chrome(헤드리스)을 띄워 가짜 Gemini 페이지를 제어한다.

Chrome 이 없는 환경에서는 건너뛴다. 실제 Gemini 화면 구조는 여기서 검증되지 않는다 (--web-probe 로 확인).
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

from council.backends import BackendError, ResponseTimeout
from council.chrome import ChromeError, find_chrome
from council.config import CouncilConfig
from council.orchestrator import Orchestrator, StopReason
from council.roles import DEFAULT_ROLES

import importlib.util

try:
    find_chrome()
    HAVE = importlib.util.find_spec("playwright") is not None
except ChromeError:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="Chrome 또는 Python Playwright 없음")

PAGE = Path(__file__).with_name("fake_gemini_page.html").resolve().as_uri()
P = DEFAULT_ROLES[0]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(tmp_path, query="", **kw):
    from council.backends.gemini_web import WebConfig
    extra = ("--headless=new",)
    if sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0:
        extra += ("--no-sandbox",)  # root 로 도는 컨테이너/CI 에서만 필요
    base = dict(port=_free_port(), profile_dir=tmp_path / "profile", tabs_file=tmp_path / "tabs.json",
                selectors_file=None, new_chat_url=PAGE + (f"?{query}" if query else ""),
                turn_interval_s=0, response_timeout_s=30, stable_s=0.4, poll_s=0.05, extra_args=extra)
    base.update(kw)
    return WebConfig(**base)


@pytest.fixture
def backends():
    made = []
    yield made
    for be in made:
        be.close()
        if be.chrome_proc is not None:
            be.chrome_proc.kill()
            be.chrome_proc.wait(timeout=10)


def _backend(backends, cfg, **kw):
    from council.backends.gemini_web import GeminiWebBackend
    be = GeminiWebBackend(cfg, log=lambda s: None, **kw)
    backends.append(be)
    return be


def test_full_discussion_then_reuse_chats(tmp_path, backends):
    cfg = _cfg(tmp_path, "end=2")
    be = _backend(backends, cfg)
    res = Orchestrator(be, "웹 자동 시험", CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")).run()
    assert res.reason is StopReason.COMPLETED and len(res.utterances) == 8
    assert all("가짜 응답" in u.text or "요약 #" in u.text for u in res.utterances)
    assert res.utterances[-1].text.endswith("[DISCUSSION_COMPLETE]")
    md = res.log_path.read_text(encoding="utf-8")
    assert "가짜 초록" in md and md.index("## 초록") < md.index("## 라운드 1")   # 초록도 웹으로
    urls = json.loads(cfg.tabs_file.read_text(encoding="utf-8"))
    assert set(urls) == {r.key for r in DEFAULT_ROLES} and all("#c" in u for u in urls.values())

    # 두 번째 실행: 저장된 채팅 주소의 탭을 다시 잡는다 (같은 Chrome 재사용, run() 이 첫 연결은 끊었음)
    cfg2 = _cfg(tmp_path, "end=1", port=cfg.port)
    be2 = _backend(backends, cfg2)
    be2.open(DEFAULT_ROLES)
    assert {k: p.url for k, p in be2._pages.items()} == urls
    assert be2.chrome_proc is None   # 이미 떠 있는 Chrome 에 붙음
    # 각 탭: 첫 메시지만 리셋 문구 + 규칙, 이후는 새 발언만
    queries = be2._pages["critic"].locator("user-query").all_inner_texts()
    assert queries[0].startswith("[새 토론 시작") and "[토론 규칙]" in queries[0]
    assert not queries[1].startswith("[새 토론 시작") and "[토론 규칙]" not in queries[1]


def test_flicker_of_stop_button_does_not_cut_answer(tmp_path, backends):
    be = _backend(backends, _cfg(tmp_path, "mode=flicker&delay=40"))
    be.open(DEFAULT_ROLES)
    reply = be.send(P, "시험 질문")
    assert reply.endswith("(3) 제안: 다음 단계") and reply.count("근거 문장") == 20


@pytest.mark.parametrize("mode,exc,needle", [
    ("captcha", BackendError, "CAPTCHA"),
    ("logout", BackendError, "입력창을 찾을 수 없음"),
])
def test_blocked_pages_stop_immediately(tmp_path, backends, mode, exc, needle):
    be = _backend(backends, _cfg(tmp_path, f"mode={mode}", page_load_timeout_s=3))
    with pytest.raises(exc, match=needle):
        be.open(DEFAULT_ROLES)


def test_usage_limit_detected(tmp_path, backends):
    be = _backend(backends, _cfg(tmp_path, "mode=limit"))
    res = Orchestrator(be, "한도", CouncilConfig(max_rounds=2, log_dir=tmp_path / "logs")).run()
    assert res.reason is StopReason.USAGE_LIMIT and res.utterances == ()
    assert "## 이어서 하려면" in res.log_path.read_text(encoding="utf-8")


def test_hang_times_out(tmp_path, backends):
    be = _backend(backends, _cfg(tmp_path, "mode=hang", response_timeout_s=2))
    be.open(DEFAULT_ROLES)
    t0 = time.monotonic()
    with pytest.raises(ResponseTimeout):
        be.send(P, "x")
    assert time.monotonic() - t0 < 10


def test_interval_between_messages(tmp_path, backends):
    be = _backend(backends, _cfg(tmp_path, turn_interval_s=3.0))
    be.open(DEFAULT_ROLES)
    t0 = time.monotonic()
    be.send(DEFAULT_ROLES[0], "첫 질문")
    t1 = time.monotonic()
    be.send(DEFAULT_ROLES[1], "둘째 질문")
    assert t1 - t0 < 3.0                      # 첫 메시지는 대기 없음
    sent = [t.answer_s for t in be.turns]
    assert len(sent) == 2
    assert time.monotonic() - t0 >= 3.0       # 둘째 메시지는 첫 전송 후 3 s 이후


def test_selectors_override(tmp_path):
    from council.backends.gemini_web import load_selectors
    f = tmp_path / "web_selectors.json"
    f.write_text(json.dumps({"_comment": "x", "input": ["#my-input"]}), encoding="utf-8")
    sel = load_selectors(f)
    assert sel["input"] == ["#my-input"] and sel["send"]          # 나머지는 기본값 유지
    f.write_text(json.dumps({"inptu": ["오타"]}), encoding="utf-8")
    with pytest.raises(BackendError, match="알 수 없는 항목"):
        load_selectors(f)
