"""Deep Research 대기열: 주제 목록을 차례로 Gemini Deep Research 로 돌리고 보고서를 .md 로 저장한다.

주제 하나의 흐름 (2026-11 사용자 화면 캡처 기준, 실제 구조는 --probe 로 확인):
  새 채팅 → 모델 버튼('Hydrogen Thinking ⌄')에서 모드 선택 → '+' → '도구 더보기' → 'Deep Research'
  → 주제 입력·전송 → 연구 계획이 나오면 '연구 시작' 클릭 → 보고서 완성까지 대기 → 보고서 글자 저장

하지 않는 것: 자동화 흔적 숨기기, 무작위 지연·마우스 흉내 등 사람 흉내, CAPTCHA 풀기.
주제 사이 간격은 고정값이다. CAPTCHA·로그아웃·사용량 한도가 보이면 대기열을 저장하고 멈춘다.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backends.base import BackendError, UsageLimitReached, UserAbort
from .backends.gemini_web import GeminiWebBackend, WebConfig
from .roles import Role
from .session import write_atomic

MODES = ("일반", "사고")
MODE_ALIASES = {"일반": "일반", "빠른": "일반", "flash": "일반", "fast": "일반", "normal": "일반",
                "사고": "사고", "thinking": "사고", "think": "사고"}
PRO_WORDS = ("pro", "프로")

# 요소 찾기 규칙 (research_selectors.json 으로 덮어쓰기 가능). 글자 기반 규칙이 많아 화면 디자인 변경에 비교적 강하다.
DEFAULT_RESEARCH_SELECTORS: dict[str, Any] = {
    "tools_button": ["button[aria-label*='업로드']", "button[aria-label*='도구']", "button[aria-label*='Upload']",
                     "button[aria-label*='tools']"],
    "more_tools_texts": ["도구 더보기", "More tools"],
    "deep_research_texts": ["Deep Research"],
    # 실제 화면 진단(2026-11-13)에서 아래 button 규칙으로는 못 찾음 → button 이 아닌 요소일 가능성. 후보를 넓히고
    # 그래도 없으면 mode_words 글자로 찾는다. 정확한 구조는 --probe 의 mode_candidates 로 확인.
    "model_button": ["[data-test-id='bard-mode-menu-button']", "bard-mode-switcher button", "bard-mode-switcher",
                     "button[aria-label*='모드']", "button[aria-label*='mode']", "button[aria-label*='모델']",
                     "button:has-text('Thinking')", "button:has-text('Flash')",
                     "[role='button']:has-text('Thinking')", "[role='button']:has-text('Flash')",
                     "[aria-haspopup]:has-text('Thinking')", "[aria-haspopup]:has-text('Flash')"],
    "mode_words": ["Thinking", "Hydrogen", "Flash", "사고", "빠른"],
    # 실제 화면(2026-11-13 진단): 모델 메뉴 = '일상생활 지원 / 4 Hydrogen', '복잡한 문제 해결 / 4 Hydrogen Thinking',
    # '상세한 조사 / 3.1 Pro'. 모델 버튼 글자 = 'Hydrogen Thinking' (일반 모드면 'Hydrogen' 으로 추정).
    # mode_texts: 메뉴에서 누를 항목 이름 (앞에서부터 시도)
    "mode_texts": {"일반": ["일상생활 지원", "Flash", "빠른"], "사고": ["복잡한 문제 해결", "Thinking", "사고"]},
    # mode_button_patterns: 선택 뒤 모델 버튼 글자로 확인 (정규식). 'Hydrogen' 은 'Hydrogen Thinking' 에도 들어
    # 있으므로 일반은 'Thinking 이 뒤따르지 않는 Hydrogen' 으로 판정한다.
    "mode_button_patterns": {"일반": r"Hydrogen(?!\s*Thinking)|Flash|빠른", "사고": r"Thinking|사고"},
    "start_texts": ["연구 시작", "Start research"],
    "done_texts": ["Docs로 내보내기", "Export to Docs", "공유 및 내보내기"],
    "report": ["deep-research-immersive-panel .markdown", "deep-research-immersive-panel", "immersive-panel",
               "[data-test-id*='immersive']"],
    "limit_texts": ["한도에 도달", "사용 한도", "reached your limit", "limit reached"],
}

LIT_TEMPLATE = """

[작성 요구]
- 선행연구를 표(연구 / 방법 / 데이터·플랫폼 / 결과 / 한계)로 정리하고, 각 행에 출처 링크를 단다.
- 확인하지 못한 주장·수치는 '확인 필요'로 표시하고 지어내지 않는다.
- 마지막에 '연구 공백(아직 안 된 것)'과 '이 주제의 차별점 후보'를 따로 정리한다."""


class TopicError(ValueError):
    """주제 목록 형식 오류."""


@dataclass(frozen=True)
class Topic:
    mode: str
    text: str

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.mode}|{self.text}".encode("utf-8")).hexdigest()[:12]


def parse_topics(text: str, default_mode: str = "사고") -> list[Topic]:
    """'모드 | 주제' 한 줄에 하나. 모드 생략 시 default_mode. '#' 으로 시작하는 줄과 빈 줄은 무시."""
    topics: list[Topic] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        mode, body = default_mode, line
        if "|" in line:
            head, body = (s.strip() for s in line.split("|", 1))
            key = head.lower()
            if key in PRO_WORDS:
                raise TopicError(f"{n}번째 줄: Pro 모델은 Deep Research 를 지원하지 않음 (일반 또는 사고로 바꿀 것)")
            if key not in MODE_ALIASES:
                raise TopicError(f"{n}번째 줄: 모드 '{head}' 를 모름 (일반 / 사고 중 하나)")
            mode = MODE_ALIASES[key]
        if not body:
            raise TopicError(f"{n}번째 줄: 주제가 비어 있음")
        topics.append(Topic(mode, body))
    return topics


# ------------------------------------------------------------------ 대기열 상태 (research_queue.json)

def load_queue(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_queue(path: Path, queue: dict[str, dict]) -> None:
    write_atomic(path, json.dumps(queue, ensure_ascii=False, indent=2))


def pending(topics: list[Topic], queue: dict[str, dict]) -> list[Topic]:
    """아직 끝나지 않은 주제 (완료 'done' 은 제외, 실패 'failed' 는 다시 시도)."""
    return [t for t in topics if queue.get(t.key, {}).get("status") != "done"]


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", text).strip("_")  # Windows 파일 이름 금지 문자 제거
    return s[:n] or "topic"


# ------------------------------------------------------------------ 실행

@dataclass
class ResearchConfig:
    """시간 단위 [s]. interval_s: 주제 사이 고정 간격 (직전 연구가 끝난 시각 기준)."""

    interval_s: float = 12 * 60.0
    plan_timeout_s: float = 60 * 60.0     # '연구 시작' 버튼 대기 (5분 → 60분: 5분 만에 실패 판정된 경험, 사용자 요청)
    research_timeout_s: float = 180 * 60.0
    stable_s: float = 120.0          # 중지 버튼이 없고 보고서가 이만큼 그대로면 완료
    done_stable_s: float = 15.0      # '내보내기' 버튼이 보이면 이만큼만 그대로여도 완료
    poll_s: float = 2.0
    countdown_s: float = 60.0
    out_dir: Path = field(default_factory=lambda: Path("research"))
    queue_file: Path = field(default_factory=lambda: Path("research_queue.json"))
    selectors_file: Path | None = field(default_factory=lambda: Path("research_selectors.json"))
    template: bool = True


def load_research_selectors(override: Path | None) -> dict[str, Any]:
    sel = json.loads(json.dumps(DEFAULT_RESEARCH_SELECTORS))
    if override is not None and override.exists():
        user = json.loads(override.read_text(encoding="utf-8"))
        unknown = set(user) - set(sel) - {"_comment"}
        if unknown:
            raise BackendError(f"research_selectors.json 에 알 수 없는 항목: {sorted(unknown)}")
        sel.update({k: v for k, v in user.items() if k != "_comment"})
    return sel


class ResearchRunner:
    """웹 백엔드(Chrome 연결, 입력창·차단 화면 감지)를 재사용해 Deep Research 를 진행한다."""

    ROLE = Role("research", "Deep Research", "-")

    def __init__(self, web: WebConfig, cfg: ResearchConfig, log: Callable[[str], None] = print,
                 commands: Callable[[], str | None] = lambda: None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = datetime.now) -> None:
        self.web = GeminiWebBackend(web, log=log, commands=commands, sleep=sleep, clock=clock)
        self.cfg, self.log, self.commands = cfg, log, commands
        self._sleep, self._clock, self._now = sleep, clock, now
        self.sel = load_research_selectors(cfg.selectors_file)
        self.ctx: Any = None

    # -------------------------------------------------------------- 공통 도우미

    def _visible(self, page: Any, css_list: list[str]) -> Any | None:
        for css in css_list:
            loc = page.locator(css)
            for i in range(loc.count() - 1, -1, -1):
                if loc.nth(i).is_visible():
                    return loc.nth(i)
        return None

    def _visible_text(self, page: Any, texts: list[str], exact: bool = False) -> Any | None:
        for t in texts:
            loc = page.get_by_text(t, exact=exact)
            for i in range(loc.count() - 1, -1, -1):
                if loc.nth(i).is_visible():
                    return loc.nth(i)
        return None

    def _menu_item(self, page: Any, texts: list[str]) -> Any | None:
        """열린 메뉴의 항목. 접근성 역할(menuitem 등)로 먼저 찾아, 같은 글자가 들어간 다른 버튼(예: 모델 버튼
        'Hydrogen Thinking')을 다시 누르는 실수를 피한다. 못 찾으면 글자로 찾는다."""
        for role in ("menuitemradio", "menuitem", "menuitemcheckbox", "option"):
            for t in texts:
                loc = page.get_by_role(role, name=re.compile(re.escape(t)))
                for i in range(loc.count() - 1, -1, -1):
                    if loc.nth(i).is_visible():
                        return loc.nth(i)
        return self._visible_text(page, texts)

    def _check(self, page: Any) -> None:
        self.web._check_blocked(page)
        if self.commands() == "q":
            raise UserAbort("사용자가 q 로 중단")

    def _last_response_text(self, page: Any) -> str:
        resp = self.web._responses(page)
        return self.web._response_text(resp.last).strip() if resp.count() else ""

    def _report_text(self, page: Any) -> str:
        loc = self._visible(page, self.sel["report"])
        if loc is not None:
            text = loc.inner_text().strip()
            if text:
                return text
        return self._last_response_text(page)

    # -------------------------------------------------------------- 단계

    MODEL_WAIT_S = 10.0  # 모델 표시는 입력창보다 늦게 그려질 수 있음 (실제 진단: 입력창 직후엔 없음)

    def _find_model_once(self, page: Any) -> Any | None:
        btn = self._visible(page, self.sel["model_button"])
        if btn is not None:
            return btn
        words = re.compile("|".join(re.escape(w) for w in self.sel["mode_words"]))
        for role in ("button", "combobox", "listbox", "menuitem"):   # 접근성 이름 (웹 컴포넌트 내부까지 닿음)
            loc = page.get_by_role(role, name=words)
            for i in range(loc.count() - 1, -1, -1):
                if loc.nth(i).is_visible():
                    return loc.nth(i)
        for w in self.sel["mode_words"]:                              # 글자 (열린 shadow DOM 포함)
            loc = page.get_by_text(w)
            for i in range(loc.count() - 1, -1, -1):
                el = loc.nth(i)
                if el.is_visible() and len(el.inner_text()) <= 40:
                    return el
        return None

    def find_model_button(self, page: Any, wait_s: float | None = None) -> Any | None:
        """모델 선택 요소. 규칙 → 접근성 역할·이름 → 글자 순으로 찾고, 늦게 그려지는 경우를 위해 잠시 기다린다.
        글자를 클릭해도 이벤트가 감싼 버튼으로 전달된다. 새 채팅에서만 쓰므로 대화 본문과 섞이지 않는다."""
        deadline = self._clock() + (self.MODEL_WAIT_S if wait_s is None else wait_s)
        while True:
            found = self._find_model_once(page)
            if found is not None or self._clock() >= deadline:
                return found
            self._sleep(0.5)

    def select_mode(self, page: Any, mode: str) -> None:
        wanted = self.sel["mode_texts"][mode]
        is_mode = re.compile(self.sel["mode_button_patterns"][mode])
        btn = self.find_model_button(page)
        if btn is None:
            raise BackendError("모델 선택 버튼을 찾을 수 없음 (--probe 로 화면 구조 확인)")
        if is_mode.search(btn.inner_text()):
            return  # 이미 원하는 모드
        btn.click()
        item = self._menu_item(page, wanted)
        if item is None:
            page.keyboard.press("Escape")
            raise BackendError(f"모델 메뉴에서 '{mode}' 항목({', '.join(wanted)})을 찾을 수 없음 (--probe 로 메뉴 이름 확인)")
        item.click()
        page.wait_for_timeout(300)  # 메뉴가 닫히고 표시가 바뀔 시간
        btn = self.find_model_button(page, wait_s=3.0)
        if btn is None or not is_mode.search(btn.inner_text()):
            raise BackendError(f"모델을 '{mode}' 로 바꾸지 못함 (현재: {btn.inner_text() if btn else '?'})")

    def enable_deep_research(self, page: Any) -> None:
        tools = self._visible(page, self.sel["tools_button"])
        if tools is None:
            raise BackendError("입력창 왼쪽 '+' 버튼을 찾을 수 없음 (--probe 로 화면 구조 확인)")
        tools.click()
        item = self._menu_item(page, self.sel["deep_research_texts"])
        if item is None:  # 하위 메뉴 '도구 더보기' 안에 있음
            more = self._menu_item(page, self.sel["more_tools_texts"])
            if more is None:
                page.keyboard.press("Escape")
                raise BackendError("'도구 더보기' 메뉴를 찾을 수 없음")
            more.click()
            item = self._menu_item(page, self.sel["deep_research_texts"])
        if item is None:
            page.keyboard.press("Escape")
            raise BackendError("'Deep Research' 항목을 찾을 수 없음 (이 모델에서 지원하지 않을 수 있음)")
        item.click()

    def _wait_plan_and_start(self, page: Any, before: int) -> None:
        deadline = self._clock() + self.cfg.plan_timeout_s
        while True:
            self._check(page)
            if self.web._responses(page).count() > before:
                text = self._last_response_text(page)
                if text and any(t in text for t in self.sel["limit_texts"]) and len(text) < 300:
                    raise UsageLimitReached(f"Deep Research 한도: {text[:200]}")
                start = self._visible_text(page, self.sel["start_texts"], exact=True)
                if start is not None:
                    start.click()
                    self.log("   '연구 시작'을 눌렀습니다. 보고서 완성까지 기다립니다.")
                    return
            if self._clock() > deadline:
                raise BackendError(f"{self.cfg.plan_timeout_s / 60:.0f}분 안에 '연구 시작' 버튼이 나오지 않음")
            self._sleep(min(self.cfg.poll_s, 1.0))

    def _wait_report(self, page: Any) -> tuple[str, bool]:
        """(보고서, 완료 여부). 시간 초과면 그때까지의 글자와 False."""
        t0 = self._clock()
        deadline = t0 + self.cfg.research_timeout_s
        last, since, last_note = "", self._clock(), t0
        while True:
            self._check(page)
            text = self._report_text(page)
            generating = self.web._first_visible(page, "stop") is not None
            now = self._clock()
            if text != last or generating or not text:
                last, since = text, now
            else:
                done_btn = self._visible_text(page, self.sel["done_texts"]) is not None
                if now - since >= (self.cfg.done_stable_s if done_btn else self.cfg.stable_s):
                    return text, True
            if now - last_note >= 10 * 60:
                last_note = now
                self.log(f"   … 연구 진행 중 ({(now - t0) / 60:.0f}분 경과, 현재 {len(text)}자)")
            if now > deadline:
                return text, False
            self._sleep(self.cfg.poll_s)

    def run_one(self, topic: Topic) -> dict:
        """주제 하나를 끝까지 진행하고 결과 기록(dict)을 돌려준다. 차단·한도는 예외로 올린다."""
        page = self.ctx.new_page()
        page.goto(self.web.cfg.new_chat_url, timeout=self.web.cfg.page_load_timeout_s * 1000)
        self.web._tabs = {self.ROLE.key: 1}
        self.web._wait_ready(page, self.ROLE)
        started = self._now()
        page.bring_to_front()
        self.select_mode(page, topic.mode)
        self.enable_deep_research(page)
        box = self.web._first_visible(page, "input")
        before = self.web._responses(page).count()
        box.click()
        box.fill(topic.text + (LIT_TEMPLATE if self.cfg.template else ""))
        send = self.web._first_visible(page, "send")
        (send.click() if send is not None and send.is_enabled() else box.press("Enter"))
        self.log(f"   주제 전송 ({topic.mode} + Deep Research). 연구 계획을 기다립니다…")
        self._wait_plan_and_start(page, before)
        report, complete = self._wait_report(page)
        finished = self._now()
        path = self._save_report(topic, report, complete, started, finished, page.url)
        return {"status": "done" if complete else "failed", "file": str(path), "chat_url": page.url,
                "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
                "chars": len(report), "error": "" if complete else f"{self.cfg.research_timeout_s / 60:.0f}분 초과 (부분 저장)"}

    def _save_report(self, topic: Topic, report: str, complete: bool, started: datetime, finished: datetime,
                     url: str) -> Path:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.out_dir / f"{started:%Y%m%d_%H%M%S}_{slugify(topic.text)}.md"
        mins = (finished - started).total_seconds() / 60
        head = [f"# Deep Research: {topic.text}", "",
                f"- 모드: {topic.mode} + Deep Research",
                f"- 시작: {started:%Y-%m-%d %H:%M:%S} / 종료: {finished:%Y-%m-%d %H:%M:%S} ({mins:.0f}분)",
                f"- 상태: {'완료' if complete else '시간 초과 — 생성 중이던 부분까지만 저장'}",
                f"- 채팅: {url}", "",
                "_Gemini Deep Research 결과를 그대로 저장. 출처 링크와 '확인 필요' 표시를 직접 확인할 것._", "", "---", ""]
        write_atomic(path, "\n".join(head) + report.strip() + "\n")
        return path

    # -------------------------------------------------------------- 대기열

    def _wait_interval(self) -> None:
        deadline = self._clock() + self.cfg.interval_s
        last_note = None
        while self._clock() < deadline:
            cmd = self.commands()
            if cmd == "q":
                raise UserAbort("사용자가 q 로 중단")
            if cmd == "n":
                self.log("   대기를 건너뜁니다.")
                return
            now = self._clock()
            if last_note is None or now - last_note >= self.cfg.countdown_s:
                last_note = now
                rem = int(round(deadline - now))
                self.log(f"⏸ 다음 주제까지 {rem // 60}분 {rem % 60:02d}초 (n + Enter: 바로 진행, q + Enter: 종료)")
            self._sleep(min(self.cfg.poll_s, max(deadline - now, 0.0)))

    def run_queue(self, topics: list[Topic]) -> dict[str, int]:
        """남은 주제를 차례로 진행. 반환: {'done', 'failed', 'left'} 개수. 대기열 파일은 주제마다 갱신."""
        queue = load_queue(self.cfg.queue_file)
        todo = pending(topics, queue)
        self.log(f"Deep Research 대기열: 전체 {len(topics)}개, 남은 주제 {len(todo)}개")
        counts = {"done": 0, "failed": 0, "left": len(todo)}
        if not todo:
            return counts
        self.ctx = self.web.connect()
        try:
            for k, topic in enumerate(todo):
                if k > 0:
                    self._wait_interval()
                self.log(f"[{k + 1}/{len(todo)}] {topic.mode} | {topic.text}")
                entry = queue.setdefault(topic.key, {"topic": topic.text, "mode": topic.mode})
                try:
                    result = self.run_one(topic)
                except BackendError as e:
                    if isinstance(e, (UsageLimitReached, UserAbort)) or "CAPTCHA" in str(e) or "로그인" in str(e):
                        entry.update(status="pending", error=f"{type(e).__name__}: {e}")
                        save_queue(self.cfg.queue_file, queue)
                        self.log(f"중단: {e} → 남은 주제는 다음 실행에서 이어 합니다.")
                        return counts
                    result = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                entry.update(result)
                save_queue(self.cfg.queue_file, queue)
                counts[result["status"]] += 1
                counts["left"] -= 1
                note = f"저장: {result['file']}" if result.get("file") else result.get("error", "")
                self.log(f"   → {('완료' if result['status'] == 'done' else '실패')}. {note}")
        finally:
            self.web.close()
        return counts
