"""Deep Research 대기열 테스트. 순수 로직 + 실제 Chrome(헤드리스)으로 가짜 Deep Research 페이지 제어."""
import importlib.util
import json
import os
import socket
import sys

import pytest

from council.chrome import ChromeError, find_chrome
from council.research import (ResearchConfig, Topic, TopicError, load_queue, parse_topics, pending, save_queue,
                              slugify)

PAGE = (__import__("pathlib").Path(__file__).with_name("fake_research_page.html").resolve().as_uri())

try:
    find_chrome()
    HAVE = importlib.util.find_spec("playwright") is not None
except ChromeError:
    HAVE = False
needs_chrome = pytest.mark.skipif(not HAVE, reason="Chrome 또는 Python Playwright 없음")


# ---------- 순수 로직 ----------

def test_parse_topics_modes_and_errors():
    text = """# 주석
사고 | IR·LiDAR 융합 선행연구
일반|열화상 깊이 추정

flash | 레이더 내비게이션
기본 모드 주제"""
    ts = parse_topics(text, default_mode="사고")
    assert [(t.mode, t.text) for t in ts] == [("사고", "IR·LiDAR 융합 선행연구"), ("일반", "열화상 깊이 추정"),
                                             ("일반", "레이더 내비게이션"), ("사고", "기본 모드 주제")]
    with pytest.raises(TopicError, match="Pro 모델은 Deep Research"):
        parse_topics("pro | 주제")
    with pytest.raises(TopicError, match="모드 '빠름'"):
        parse_topics("빠름 | 주제")
    with pytest.raises(TopicError, match="비어"):
        parse_topics("사고 | ")


def test_queue_skips_done_and_retries_failed(tmp_path):
    ts = [Topic("사고", "A"), Topic("일반", "B"), Topic("사고", "C")]
    q = {ts[0].key: {"status": "done"}, ts[1].key: {"status": "failed"}}
    save_queue(tmp_path / "q.json", q)
    assert pending(ts, load_queue(tmp_path / "q.json")) == ts[1:]
    assert Topic("사고", "A").key != Topic("일반", "A").key        # 같은 주제라도 모드가 다르면 별개


def test_slugify_windows_safe():
    assert slugify('IR/LiDAR: "융합" 연구?') == "IR_LiDAR_융합_연구"
    assert slugify("???") == "topic" and len(slugify("가" * 100)) == 40
    assert not any(c in slugify('a\\b/c:d*e?f"g<h>i|j k') for c in '\\/:*?"<>| ')


# ---------- 실제 Chrome + 가짜 페이지 ----------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _runner(tmp_path, query="", **kw):
    from council.backends.gemini_web import WebConfig
    from council.research import ResearchRunner
    extra = ("--headless=new",)
    if sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0:
        extra += ("--no-sandbox",)
    web = WebConfig(port=_free_port(), profile_dir=tmp_path / "profile", tabs_file=tmp_path / "tabs.json",
                    selectors_file=None, new_chat_url=PAGE + (f"?{query}" if query else ""), poll_s=0.05,
                    page_load_timeout_s=5, extra_args=extra)
    base = dict(interval_s=0, plan_timeout_s=10, research_timeout_s=30, stable_s=1.0, done_stable_s=0.3,
                poll_s=0.1, out_dir=tmp_path / "research", queue_file=tmp_path / "queue.json", selectors_file=None)
    base.update(kw)
    logs = []
    r = ResearchRunner(web, ResearchConfig(**base), log=logs.append)
    return r, logs


@pytest.fixture
def cleanup():
    made = []
    yield made
    for r in made:
        r.web.close()
        if r.web.chrome_proc is not None:
            r.web.chrome_proc.kill()
            r.web.chrome_proc.wait(timeout=10)


@needs_chrome
def test_queue_runs_topics_end_to_end(tmp_path, cleanup):
    r, logs = _runner(tmp_path)
    cleanup.append(r)
    topics = [Topic("일반", "열화상 깊이 추정"), Topic("사고", "레이더 내비게이션")]
    counts = r.run_queue(topics)
    assert counts == {"done": 2, "failed": 0, "left": 0}
    q = load_queue(tmp_path / "queue.json")
    files = sorted((tmp_path / "research").glob("*.md"))
    assert len(files) == 2 and all(q[t.key]["status"] == "done" for t in topics)
    text = files[0].read_text(encoding="utf-8")
    assert "선행연구 대조표" in text and "가짜 보고서 끝 ✓" in text and "모드: " in text
    assert any("[1/2]" in m for m in logs) and any("연구 시작" in m for m in logs)
    # 두 번째 실행: 모두 완료됐으므로 아무것도 하지 않음
    r2, _ = _runner(tmp_path)
    cleanup.append(r2)
    assert r2.run_queue(topics) == {"done": 0, "failed": 0, "left": 0}


@needs_chrome
def test_sent_message_has_mode_deep_research_and_template(tmp_path, cleanup):
    """가짜 페이지는 받은 메시지 앞에 [모델 + Deep Research] 상태를 기록한다 → 실제로 반영됐는지 확인."""
    r, _ = _runner(tmp_path)
    cleanup.append(r)
    r.ctx = r.web.connect()
    result = r.run_one(Topic("일반", "열화상 깊이 추정"))
    sent = r.ctx.pages[-1].locator("user-query").first.inner_text()
    assert sent.startswith("[Hydrogen Flash + Deep Research] 열화상 깊이 추정") and "[작성 요구]" in sent
    assert result["status"] == "done" and result["chat_url"].startswith(PAGE)
    r.cfg.template = False
    r.run_one(Topic("사고", "원문만"))
    sent2 = r.ctx.pages[-1].locator("user-query").first.inner_text()
    assert sent2.startswith("[Hydrogen Thinking + Deep Research] 원문만") and "[작성 요구]" not in sent2


@needs_chrome
def test_mode_and_tools_actually_selected(tmp_path, cleanup):
    r, _ = _runner(tmp_path)
    cleanup.append(r)
    r.ctx = r.web.connect()
    page = r.ctx.new_page()
    page.goto(PAGE)
    r.select_mode(page, "일반")
    assert page.locator("#model-label").inner_text() == "Hydrogen Flash"
    r.select_mode(page, "사고")
    assert page.locator("#model-label").inner_text() == "Hydrogen Thinking"
    r.enable_deep_research(page)
    assert page.locator("#chip").is_visible()


@needs_chrome
def test_limit_stops_queue_and_keeps_pending(tmp_path, cleanup):
    r, logs = _runner(tmp_path, "mode=limit")
    cleanup.append(r)
    topics = [Topic("사고", "A"), Topic("사고", "B")]
    counts = r.run_queue(topics)
    assert counts == {"done": 0, "failed": 0, "left": 2}
    q = load_queue(tmp_path / "queue.json")
    assert q[topics[0].key]["status"] == "pending" and "한도" in q[topics[0].key]["error"]
    assert pending(topics, q) == topics                           # 다음 실행에서 둘 다 다시 시도


@needs_chrome
def test_no_plan_marks_failed_and_continues(tmp_path, cleanup):
    r, _ = _runner(tmp_path, "mode=noplan", plan_timeout_s=2)
    cleanup.append(r)
    counts = r.run_queue([Topic("사고", "A"), Topic("일반", "B")])
    assert counts == {"done": 0, "failed": 2, "left": 0}
    q = load_queue(tmp_path / "queue.json")
    assert all("연구 시작" in e["error"] for e in q.values())


@needs_chrome
def test_research_timeout_saves_partial(tmp_path, cleanup):
    r, _ = _runner(tmp_path, "mode=hang", research_timeout_s=3)
    cleanup.append(r)
    counts = r.run_queue([Topic("사고", "멈추는 연구")])
    assert counts["failed"] == 1
    md = next((tmp_path / "research").glob("*.md")).read_text(encoding="utf-8")
    assert "시간 초과" in md


@needs_chrome
def test_captcha_stops_queue(tmp_path, cleanup):
    r, logs = _runner(tmp_path, "mode=captcha")
    cleanup.append(r)
    counts = r.run_queue([Topic("사고", "A")])
    assert counts == {"done": 0, "failed": 0, "left": 1}
    assert json.loads((tmp_path / "queue.json").read_text(encoding="utf-8")).popitem()[1]["status"] == "pending"


def test_topics_file_with_notepad_bom(tmp_path):
    """메모장이 붙이는 UTF-8 BOM 이 있어도 첫 줄 주석이 주제로 읽히지 않아야 함."""
    from council.research import parse_topics
    f = tmp_path / "research_topics.txt"
    f.write_bytes("# 주석\n사고 | 주제 A\n".encode("utf-8-sig"))
    assert [t.text for t in parse_topics(f.read_text(encoding="utf-8-sig"))] == ["주제 A"]
    assert len(parse_topics(f.read_text(encoding="utf-8"))) == 2   # BOM 을 안 벗기면 주석이 주제가 됨 (회귀 기준)


def test_chrome_launch_failure_is_clean_message(tmp_path, monkeypatch, capsys):
    from council import research_cli
    monkeypatch.setenv("COUNCIL_CHROME", str(tmp_path / "없는_chrome.exe"))
    f = tmp_path / "t.txt"
    f.write_text("사고 | 주제\n", encoding="utf-8")
    (tmp_path / "profile").mkdir()
    rc = research_cli.run(["--topics", str(f), "--profile-dir", str(tmp_path / "profile"), "--port", "9599",
                           "--out-dir", str(tmp_path / "r")], None)
    assert rc == 1 and "시작할 수 없음" in capsys.readouterr().out
