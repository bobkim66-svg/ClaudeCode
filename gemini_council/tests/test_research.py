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
                poll_s=0.1, out_dir=tmp_path / "research", queue_file=tmp_path / "queue.json", selectors_file=None,
                quota_file=tmp_path / "quota.json", wait_poll_s=1800,
                wait_on_limit=False)  # 한도 대기 테스트에서만 켠다 (기본 켜짐이면 실제로 1시간 잠듦)
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
    assert sent.startswith("[Hydrogen + Deep Research] 열화상 깊이 추정") and "[작성 요구]" in sent
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
    assert page.locator("#model-label").inner_text() == "Hydrogen"
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


@needs_chrome
def test_mode_selection_when_picker_is_not_a_button(tmp_path, cleanup):
    """실제 화면처럼 모델 선택이 버튼 규칙에 안 맞아도, 'Thinking'·'Flash' 글자로 찾아 바꾼다."""
    r, _ = _runner(tmp_path)
    cleanup.append(r)
    r.ctx = r.web.connect()
    page = r.ctx.new_page()
    page.goto(PAGE + "?picker=div")
    assert r._visible(page, r.sel["model_button"]) is None          # 규칙으로는 못 찾는 상황 재현
    r.select_mode(page, "일반")
    assert page.locator("#model-label").inner_text() == "Hydrogen"
    r.select_mode(page, "사고")
    assert page.locator("#model-label").inner_text() == "Hydrogen Thinking"


@needs_chrome
@pytest.mark.parametrize("picker", ["shadow", "late"])
def test_mode_selection_shadow_dom_or_late_render(tmp_path, cleanup, picker):
    """모델 표시가 웹 컴포넌트(shadow DOM) 안에 있거나 입력창보다 늦게 그려져도 찾아서 바꾼다."""
    r, _ = _runner(tmp_path)
    cleanup.append(r)
    r.ctx = r.web.connect()
    page = r.ctx.new_page()
    page.goto(PAGE + f"?picker={picker}")
    r.select_mode(page, "일반")
    assert page.locator("#model-label").inner_text() == "Hydrogen"   # CSS 는 열린 shadow DOM 을 통과
    r.select_mode(page, "사고")
    assert page.locator("#model-label").inner_text() == "Hydrogen Thinking"


def test_mode_button_patterns_distinguish_hydrogen_from_thinking():
    """'Hydrogen' 이 'Hydrogen Thinking' 안에도 들어 있어 단순 포함 검사는 일반/사고를 혼동한다 → 정규식으로 구분."""
    import re
    from council.research import DEFAULT_RESEARCH_SELECTORS as S
    general, think = (re.compile(S["mode_button_patterns"][m]) for m in ("일반", "사고"))
    assert general.search("Hydrogen") and not general.search("Hydrogen Thinking")
    assert think.search("Hydrogen Thinking") and not think.search("Hydrogen")


def test_probe_topic_option():
    from council.research_cli import PROBE_TOPIC, build_parser
    assert build_parser().parse_args([]).probe_topic == PROBE_TOPIC and "젠슨 황" in PROBE_TOPIC
    assert build_parser().parse_args(["--probe-topic", "다른 주제"]).probe_topic == "다른 주제"


def test_plan_wait_unlimited_by_default_and_other_waits_unchanged():
    """사용자 요청: '연구 시작' 버튼 대기 제한 없음(0), 나머지(완료 판단 2분/15초, 최대 180분)는 그대로."""
    rc = ResearchConfig()
    assert rc.plan_timeout_s == 0 and rc.stable_s == 120 and rc.done_stable_s == 15
    assert rc.research_timeout_s == 180 * 60
    from council.research_cli import build_parser, configs
    _, cli = configs(build_parser().parse_args([]))
    assert (cli.plan_timeout_s, cli.stable_s, cli.done_stable_s) == (0, 120, 15)
    _, cli2 = configs(build_parser().parse_args(["--plan-minutes", "90"]))
    assert cli2.plan_timeout_s == 5400


@needs_chrome
def test_timing_recorded_and_unlimited_plan_wait(tmp_path, cleanup):
    """계획이 늦게 나와도(제한 없음) 기다리고, 계획·연구·전체 소요 시간을 로그·대기열·보고서에 남긴다."""
    r, logs = _runner(tmp_path, "plan=2500", plan_timeout_s=0, plan_note_s=1.0)
    cleanup.append(r)
    assert r.run_queue([Topic("사고", "시간 측정")]) == {"done": 1, "failed": 0, "left": 0}
    q = load_queue(tmp_path / "queue.json")
    timing = next(iter(q.values()))["timing"]
    assert timing["plan_s"] >= 2 and timing["total_s"] >= timing["plan_s"] + timing["research_s"] - 1
    assert any("연구 계획 기다리는 중" in m for m in logs)            # 대기 중 경과 표시
    assert any("연구 계획까지" in m and "걸림" in m for m in logs)
    md = next((tmp_path / "research").glob("*.md")).read_text(encoding="utf-8")
    assert "- 소요: 계획 생성 " in md


# ---------- 한도: 문구 인식·시도 기록·다음 시도 시각 ----------

from datetime import date, datetime, timedelta, timezone  # noqa: E402

from council.quota import QuotaLog, next_try_time, pacific_midnight, pacific_offset, parse_limit_date  # noqa: E402

KST = timezone(timedelta(hours=9))
REAL_MSG = "일일 조사 요청 1개를 모두 사용하셨습니다. 새로운 연구 요청은 2026. 11. 13. 이후에 진행하실 수 있습니다."


def test_parse_limit_date_variants():
    assert parse_limit_date(REAL_MSG) == date(2026, 11, 13)
    assert parse_limit_date("요청은 2026.1.5 이후에") == date(2026, 1, 5)
    assert parse_limit_date("한도에 도달했습니다") is None
    assert parse_limit_date("2026. 13. 40. 이후") is None          # 잘못된 날짜는 None


def test_pacific_dst_boundaries():
    utc = timezone.utc
    assert pacific_offset(datetime(2026, 3, 8, 9, 59, tzinfo=utc)) == timedelta(hours=-8)
    assert pacific_offset(datetime(2026, 3, 8, 10, 0, tzinfo=utc)) == timedelta(hours=-7)
    assert pacific_offset(datetime(2026, 11, 1, 8, 59, tzinfo=utc)) == timedelta(hours=-7)
    assert pacific_offset(datetime(2026, 11, 1, 9, 0, tzinfo=utc)) == timedelta(hours=-8)
    assert pacific_midnight(date(2026, 11, 13)).astimezone(KST) == datetime(2026, 11, 13, 17, 0, tzinfo=KST)
    assert pacific_midnight(date(2026, 7, 1)).astimezone(KST) == datetime(2026, 7, 1, 16, 0, tzinfo=KST)


def test_next_try_takes_latest_candidate_and_backoff_grows():
    now = datetime(2026, 11, 13, 12, 57, tzinfo=KST)
    when, reasons = next_try_time(now, REAL_MSG, last_start=now, attempt=0)
    assert when == datetime(2026, 11, 14, 13, 27, tzinfo=KST)         # 시작 + 24시간 + 30분 (롤링 가설)
    assert sum(r.startswith("★") for r in reasons) == 1
    # 날짜가 더 늦으면 날짜(PT 자정) 쪽
    when2, _ = next_try_time(now, REAL_MSG.replace("11. 13.", "11. 16."), last_start=now, attempt=0)
    assert when2 == datetime(2026, 11, 16, 17, 30, tzinfo=KST)
    # 근거가 없으면 최소 간격 1 → 3 → 6 → 12 → 12 시간
    steps = [(next_try_time(now, "한도", None, a)[0] - now) / timedelta(hours=1) for a in range(5)]
    assert steps == [1, 3, 6, 12, 12]
    assert next_try_time(now, "주간 한도에 도달", None, 0)[0] == now + timedelta(hours=24)


def test_quota_log_state_and_reload(tmp_path):
    f = tmp_path / "q.json"
    log = QuotaLog.load(f)
    t0 = datetime(2026, 11, 13, 12, 50, tzinfo=KST)
    log.add(t0 - timedelta(days=1), "start_clicked", "어제")
    log.add(t0 - timedelta(hours=20), "done", "어제")                    # 실제 진행된 연구 → 24시간 기준
    log.add(t0, "start_clicked", "A")
    when, _ = log.on_limit(t0 + timedelta(minutes=7), "A", REAL_MSG)   # 한도로 끝난 클릭은 기준에서 뺌
    log2 = QuotaLog.load(f)                                            # 재시작 후에도 유지
    assert log2.last_start == t0 - timedelta(days=1) and "pending_start" not in log2.state
    # max(지금+1h=13:57, 어제 시작+24h+30분=13:20, 안내 날짜 PT 자정+30분=17:30) → 17:30 KST
    assert log2.next_try == when == datetime(2026, 11, 13, 17, 30, tzinfo=KST)
    assert log2.state["limit_streak"] == 1
    assert [e["event"] for e in log2.events][-2:] == ["start_clicked", "limit"] and "모두 사용" in log2.events[-1]["message"]
    log2.add(t0 + timedelta(days=1), "start_clicked", "A")
    log2.add(t0 + timedelta(days=1, hours=1), "done", "A")
    assert log2.state["limit_streak"] == 0 and log2.next_try is None and log2.last_start == t0 + timedelta(days=1)


class _FakeTime:
    """sleep 하면 벽시계(now)가 그만큼 흐르는 가짜 시간. on_sleep 으로 대기 중 상황을 바꿀 수 있다."""

    def __init__(self, start, on_sleep=None):
        self.t, self.slept, self.on_sleep = start, 0.0, on_sleep

    def now(self):
        return self.t

    def sleep(self, s):
        import time
        self.slept += s
        self.t += timedelta(seconds=s)
        time.sleep(min(s, 0.01))
        if self.on_sleep:
            self.on_sleep(self)


@needs_chrome
def test_limit_after_start_is_not_saved_as_done_and_no_wait_stops(tmp_path, cleanup):
    """실제 버그: '연구 시작' 뒤 나온 한도 안내가 '완료' 보고서로 저장됐다."""
    r, logs = _runner(tmp_path, "mode=limit_after_start")
    cleanup.append(r)
    topics = [Topic("사고", "A"), Topic("사고", "B")]
    assert r.run_queue(topics) == {"done": 0, "failed": 0, "left": 2}
    q = load_queue(tmp_path / "queue.json")
    assert q[topics[0].key]["status"] == "pending" and "모두 사용" in q[topics[0].key]["error"]
    assert not list((tmp_path / "research").glob("*.md"))              # 보고서로 저장하지 않음
    log = json.loads((tmp_path / "quota.json").read_text(encoding="utf-8"))
    assert [e["event"] for e in log["events"]] == ["sent", "start_clicked", "limit"]
    assert log["state"]["next_try"] and any("★" in m for m in logs)


@needs_chrome
def test_limit_waits_then_retries_same_topic(tmp_path, cleanup):
    r, logs = _runner(tmp_path, "mode=limit_after_start", wait_on_limit=True, max_wait_s=48 * 3600)
    cleanup.append(r)

    def lift(ft):  # 기다리는 동안 한도가 풀림 → 정상 페이지
        r.web.cfg.new_chat_url = PAGE
    ft = _FakeTime(datetime(2026, 11, 13, 12, 50, tzinfo=KST), on_sleep=lift)
    r._now, r._sleep = ft.now, ft.sleep
    topics = [Topic("사고", "A")]
    assert r.run_queue(topics) == {"done": 1, "failed": 0, "left": 0}
    # 한도 클릭은 기준에서 빠지므로 안내 날짜(11/13 PT 자정 = 17:00 KST) + 30분 = 17:30 까지 ≈ 4시간 40분
    assert 4.5 * 3600 < ft.slept < 4.8 * 3600
    events = [e["event"] for e in json.loads((tmp_path / "quota.json").read_text(encoding="utf-8"))["events"]]
    assert events == ["sent", "start_clicked", "limit", "sent", "start_clicked", "done"]
    assert any("한도 대기" in m for m in logs)


@needs_chrome
def test_limit_wait_longer_than_max_stops(tmp_path, cleanup):
    r, logs = _runner(tmp_path, "mode=limit_after_start", wait_on_limit=True, max_wait_s=3600)
    cleanup.append(r)
    ft = _FakeTime(datetime(2026, 11, 13, 12, 50, tzinfo=KST))
    r._now, r._sleep = ft.now, ft.sleep
    assert r.run_queue([Topic("사고", "A")]) == {"done": 0, "failed": 0, "left": 1}
    assert ft.slept < 60 and any("최대 대기" in m for m in logs)


def test_startup_waits_for_saved_next_try(tmp_path):
    """지난 실행에서 저장한 next_try 가 미래면 Chrome 에 연결하기 전에 기다린다 (q 로 중단 가능)."""
    from council.backends.gemini_web import WebConfig
    from council.research import ResearchRunner
    log = QuotaLog.load(tmp_path / "quota.json")
    t0 = datetime(2026, 11, 13, 12, 50, tzinfo=KST)
    log.state["next_try"] = (t0 + timedelta(hours=2)).isoformat()
    log.save()
    cmds = iter([None, None, "q"])
    ft = _FakeTime(t0)
    logs = []
    r = ResearchRunner(WebConfig(tabs_file=tmp_path / "t.json", selectors_file=None),
                       ResearchConfig(queue_file=tmp_path / "queue.json", quota_file=tmp_path / "quota.json",
                                      selectors_file=None, out_dir=tmp_path / "r", wait_poll_s=600),
                       log=logs.append, commands=lambda: next(cmds, None), sleep=ft.sleep, now=ft.now)
    assert r.run_queue([Topic("사고", "A")]) == {"done": 0, "failed": 0, "left": 1}
    assert any("기다립니다" in m for m in logs) and r.ctx is None       # 연결 전에 멈춤


@needs_chrome
def test_too_short_report_is_failed_not_done(tmp_path, cleanup):
    r, _ = _runner(tmp_path, "mode=short")
    cleanup.append(r)
    assert r.run_queue([Topic("사고", "A")]) == {"done": 0, "failed": 1, "left": 0}
    q = load_queue(tmp_path / "queue.json")
    assert "너무 짧음" in next(iter(q.values()))["error"]
