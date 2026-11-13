"""자동 모드: Playwright 로 내 Chrome(전용 프로필)의 Gemini 웹 탭 4개를 직접 제어한다.

화면 좌표·이미지 인식이 아니라 페이지 구조(DOM)를 읽고 쓴다 → 팝업·스크롤·창 크기에 덜 민감하고,
답변을 화면에서 직접 읽으므로 OS 클립보드를 쓰지 않는다 (다른 복사 작업과 충돌 없음).

하지 않는 것: 자동화 흔적 숨기기, 사람 흉내(무작위 지연 등), CAPTCHA 풀기.
CAPTCHA·비정상 트래픽·로그아웃·사용량 한도가 보이면 즉시 저장 후 멈춘다 (다음에 이어 하기).

Gemini 화면 구조는 바뀔 수 있으므로 요소 찾기 규칙(selectors)은 web_selectors.json 으로 덮어쓸 수 있다.
기본값은 실제 화면으로 검증되지 않았다 → 첫 사용 전 --web-probe 로 확인할 것.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..chrome import DEFAULT_PORT, ChromeError, cdp_endpoint, default_profile_dir, launch_chrome
from ..roles import Role
from .base import BackendError, ChatBackend, ResponseTimeout, UsageLimitReached, UserAbort

GEMINI_NEW_CHAT_URL = "https://gemini.google.com/app"

# 요소 찾기 규칙: 각 항목은 후보 목록이며 위에서부터 처음 보이는 것을 쓴다.
DEFAULT_SELECTORS: dict[str, Any] = {
    "input": ["rich-textarea div.ql-editor[contenteditable='true']",
              "div[contenteditable='true'][role='textbox']", "textarea"],
    "send": ["button.send-button", "button[aria-label*='보내기']", "button[aria-label*='Send']"],
    "stop": ["button.stop-button", "button[aria-label*='중지']", "button[aria-label*='Stop']"],
    "response_container": ["model-response", "[data-test-id='model-response']"],
    "response_text": ["message-content .markdown", "message-content", ".markdown"],
    "captcha": ["iframe[title*='reCAPTCHA']", "iframe[src*='recaptcha']", "#captcha-form"],
    "captcha_url_parts": ["/sorry/"],
    "login_url_parts": ["accounts.google.com"],
    # 답변 본문이 이런 문구면 사용량 한도로 본다 (짧은 답변일 때만 검사해 오탐을 줄인다)
    "limit_texts": ["한도에 도달", "사용량 한도", "reached your limit", "usage limit"],
    "limit_max_chars": 300,
}
CHAT_URL_RE = re.compile(r"/app/[0-9a-zA-Z]+|#c[0-9a-z]+")   # 실제 Gemini 채팅 주소 / 가짜 페이지 주소


def load_selectors(override: Path | None) -> dict[str, Any]:
    """기본 규칙에 web_selectors.json 의 항목을 덮어쓴다 (있을 때만)."""
    sel = json.loads(json.dumps(DEFAULT_SELECTORS))
    if override is not None and override.exists():
        user = json.loads(override.read_text(encoding="utf-8"))
        unknown = set(user) - set(sel) - {"_comment"}
        if unknown:
            raise BackendError(f"web_selectors.json 에 알 수 없는 항목: {sorted(unknown)}")
        sel.update({k: v for k, v in user.items() if k != "_comment"})
    return sel


@dataclass
class WebConfig:
    """자동 모드 설정. 시간 단위 [s].

    turn_interval_s: 연속한 두 메시지 사이 최소 간격 (직전 메시지를 보낸 시각 기준, 고정값).
    response_timeout_s: 메시지 1개의 답변 완료를 기다리는 최대 시간 (Pro 사고 모드 고려).
    stable_s: 중지 버튼이 사라진 뒤 답변 글자가 이만큼 변하지 않아야 완료로 본다 (중지 버튼 깜빡임 대비).
    tabs_file: 역할별 채팅 주소 저장 파일 (첫 실행에 새 채팅을 만들고 기록, 이후 재사용).
    new_chat_url: 새 채팅 주소. extra_args: Chrome 추가 옵션 (테스트의 --headless=new 등).
    """

    port: int = DEFAULT_PORT
    profile_dir: Path = field(default_factory=default_profile_dir)
    tabs_file: Path = field(default_factory=lambda: Path("tabs.json"))
    selectors_file: Path | None = field(default_factory=lambda: Path("web_selectors.json"))
    new_chat_url: str = GEMINI_NEW_CHAT_URL
    turn_interval_s: float = 16 * 60.0
    response_timeout_s: float = 600.0
    page_load_timeout_s: float = 60.0
    stable_s: float = 2.0
    poll_s: float = 0.5
    countdown_s: float = 60.0
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class WebTurn:
    role_name: str
    tab: int
    sent_chars: int
    reply_chars: int
    answer_s: float


class GeminiWebBackend(ChatBackend):
    """웹 탭 4개를 직접 제어하는 백엔드 (탭이 대화를 기억하므로 stateful)."""

    stateful = True

    def __init__(self, cfg: WebConfig | None = None, log: Callable[[str], None] = print,
                 commands: Callable[[], str | None] = lambda: None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg or WebConfig()
        self.log, self.commands, self._sleep, self._clock = log, commands, sleep, clock
        self.sel = load_selectors(self.cfg.selectors_file)
        self._pw = self._browser = None
        self.chrome_proc = None  # 이 백엔드가 직접 띄운 Chrome (이미 떠 있던 것을 재사용하면 None)
        self._pages: dict[str, Any] = {}
        self._tabs: dict[str, int] = {}
        self._urls: dict[str, str] = {}
        self._last_send: float | None = None
        self._reset_done: set[str] = set()
        self.turns: list[WebTurn] = []

    # ------------------------------------------------------------------ 연결

    def _load_urls(self) -> dict[str, str]:
        f = self.cfg.tabs_file
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8-sig"))  # 메모장으로 고친 파일의 BOM 허용
            return {k: v for k, v in data.items() if isinstance(v, str)}
        return {}

    def _save_urls(self) -> None:
        self.cfg.tabs_file.write_text(json.dumps(self._urls, ensure_ascii=False, indent=2), encoding="utf-8")

    def connect(self) -> Any:
        """Chrome 을 띄우거나(이미 떠 있으면 재사용) CDP 로 붙는다. 브라우저 컨텍스트를 돌려준다."""
        from playwright.sync_api import sync_playwright  # 무거운 모듈이라 자동 모드에서만 불러온다

        try:
            self.chrome_proc = launch_chrome(self.cfg.port, self.cfg.profile_dir, extra_args=self.cfg.extra_args)
        except ChromeError as e:  # 토론·연구 모두 '백엔드 오류'로 저장하고 알아볼 수 있는 메시지로 끝내기 위함
            raise BackendError(str(e)) from e
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.connect_over_cdp(cdp_endpoint(self.cfg.port))
        if not self._browser.contexts:
            raise BackendError("Chrome 브라우저 컨텍스트를 찾을 수 없음")
        return self._browser.contexts[0]

    def open(self, roles: tuple[Role, ...]) -> None:
        ctx = self.connect()
        self._tabs = {r.key: i + 1 for i, r in enumerate(roles)}
        self._urls = self._load_urls()
        existing = {p.url: p for p in ctx.pages}
        for r in roles:
            url = self._urls.get(r.key)
            page = existing.get(url) if url else None
            if page is None:
                page = ctx.new_page()
                page.goto(url or self.cfg.new_chat_url, timeout=self.cfg.page_load_timeout_s * 1000)
            self._pages[r.key] = page
            self._wait_ready(page, r)
            where = f"등록된 채팅 사용 (…{url[-16:]})" if url else "새 채팅 (첫 답변 후 주소 저장)"
            self.log(f"자동 모드: 탭 {self._tabs[r.key]} · {r.name} 준비 — {where}")

    def close(self) -> None:
        # 탭과 Chrome 은 그대로 둔다 (아침에 대화를 직접 볼 수 있게). 연결만 끊는다.
        if self._pw is not None:
            self._pw.stop()
            self._pw = self._browser = None

    # ------------------------------------------------------------------ 화면 상태 검사

    def _first_visible(self, page: Any, key: str) -> Any | None:
        for css in self.sel[key]:
            loc = page.locator(css)
            n = loc.count()
            for i in range(n - 1, -1, -1):  # 여러 개면 마지막(가장 최근) 것
                if loc.nth(i).is_visible():
                    return loc.nth(i)
        return None

    def _check_blocked(self, page: Any) -> None:
        url = page.url
        if any(p in url for p in self.sel["login_url_parts"]):
            raise BackendError("Gemini 로그인이 풀림. --web-login 으로 다시 로그인한 뒤 이어 하기")
        if any(p in url for p in self.sel["captcha_url_parts"]) or self._first_visible(page, "captcha"):
            raise BackendError("CAPTCHA / 비정상적인 트래픽 화면 감지 → 자동 진행 중단. "
                               "Chrome 창에서 직접 확인을 마친 뒤 exe 를 다시 실행해 이어 하기")

    def _wait_ready(self, page: Any, role: Role) -> None:
        deadline = self._clock() + self.cfg.page_load_timeout_s
        while True:
            self._check_blocked(page)
            if self._first_visible(page, "input") is not None:
                return
            if self._clock() > deadline:
                raise BackendError(f"탭 {self._tabs.get(role.key)} · {role.name}: 입력창을 찾을 수 없음 "
                                   "(로그인 안 됨 또는 Gemini 화면 구조 변경 → --web-login / --web-probe)")
            self._sleep(self.cfg.poll_s)

    def _responses(self, page: Any) -> Any:
        for css in self.sel["response_container"]:
            loc = page.locator(css)
            if loc.count() > 0:
                return loc
        return page.locator(self.sel["response_container"][0])

    def _response_text(self, container: Any) -> str:
        for css in self.sel["response_text"]:
            inner = container.locator(css)
            if inner.count() > 0:
                return inner.last.inner_text()
        return container.inner_text()

    # ------------------------------------------------------------------ 간격

    def _wait_interval(self, tab: int, role_name: str) -> None:
        if self._last_send is None or self.cfg.turn_interval_s <= 0:
            return
        deadline = self._last_send + self.cfg.turn_interval_s
        last_note = None
        while True:
            cmd = self.commands()
            if cmd == "q":
                raise UserAbort("사용자가 q 로 중단")
            if cmd == "n":
                self.log("   대기를 건너뜁니다.")
                return
            now = self._clock()
            if now >= deadline:
                return
            if last_note is None or now - last_note >= self.cfg.countdown_s:
                last_note = now
                rem = int(round(deadline - now))
                self.log(f"   ⏸ 다음 차례 [탭 {tab} · {role_name}] 까지 {rem // 60}분 {rem % 60:02d}초 "
                         "(n + Enter: 바로 진행, q + Enter: 저장 후 종료)")
            self._sleep(min(self.cfg.poll_s, max(deadline - now, 0.0)))

    # ------------------------------------------------------------------ 보내기

    def send(self, role: Role, prompt: str) -> str:
        page = self._pages.get(role.key)
        if page is None:
            raise BackendError("open() 전에 send() 호출됨")
        tab = self._tabs[role.key]
        self._wait_interval(tab, role.name)
        if role.key not in self._reset_done:
            from .manual import RESET_PREFIX  # 반자동과 같은 리셋 문구 (탭 재사용)
            prompt = RESET_PREFIX + prompt
            self._reset_done.add(role.key)
        # 백그라운드 탭은 화면 갱신이 늦춰져 클릭 준비 확인이 수 초씩 걸린다(실측) → 입력할 탭을 앞으로
        page.bring_to_front()
        self._check_blocked(page)
        before = self._responses(page).count()

        box = self._first_visible(page, "input")
        if box is None:
            raise BackendError(f"탭 {tab} · {role.name}: 입력창 없음 (--web-probe 로 화면 구조 확인)")
        box.click()
        box.fill(prompt)
        btn = self._first_visible(page, "send")
        if btn is not None and btn.is_enabled():
            btn.click()
        else:
            box.press("Enter")
        self._last_send = self._clock()
        self.log(f"   → [탭 {tab} · {role.name}] 입력 {len(prompt)}자, 답변 기다리는 중…")

        reply = self._wait_reply(page, before, tab, role)
        if role.key not in self._urls and CHAT_URL_RE.search(page.url):
            self._urls[role.key] = page.url        # 새 채팅이면 주소를 기록해 다음에 재사용
            self._save_urls()
        if len(reply) <= self.sel["limit_max_chars"] and any(t in reply for t in self.sel["limit_texts"]):
            raise UsageLimitReached(f"탭 {tab} · {role.name}: {reply[:200]}")
        self.turns.append(WebTurn(role.name, tab, len(prompt), len(reply), self._clock() - self._last_send))
        return reply

    def _wait_reply(self, page: Any, before: int, tab: int, role: Role) -> str:
        """새 답변이 생기고 + 중지 버튼이 없고 + 글자가 stable_s 동안 그대로면 완료."""
        deadline = self._clock() + self.cfg.response_timeout_s
        last_text, stable_since = None, None
        while True:
            self._check_blocked(page)
            if self.commands() == "q":
                raise UserAbort("사용자가 q 로 중단")
            resp = self._responses(page)
            if resp.count() > before:
                text = self._response_text(resp.last).strip()
                generating = self._first_visible(page, "stop") is not None
                now = self._clock()
                if text != last_text or generating or not text:
                    last_text, stable_since = text, now
                elif now - stable_since >= self.cfg.stable_s:
                    return text
            if self._clock() > deadline:
                raise ResponseTimeout(f"탭 {tab} · {role.name}: {self.cfg.response_timeout_s / 60:.0f}분 안에 "
                                      "답변이 끝나지 않음")
            self._sleep(self.cfg.poll_s)

    def usage_report(self) -> str:
        if not self.turns:
            return ""
        rows = ["| 탭 | 발언자 | 보낸 글자 | 받은 글자 | 답변 시간 [s] |", "|---:|---|---:|---:|---:|"]
        rows += [f"| {t.tab} | {t.role_name} | {t.sent_chars} | {t.reply_chars} | {t.answer_s:.0f} |"
                 for t in self.turns]
        rows.append(f"| | 합계 ({len(self.turns)}회) | {sum(t.sent_chars for t in self.turns)} "
                    f"| {sum(t.reply_chars for t in self.turns)} | {sum(t.answer_s for t in self.turns):.0f} |")
        return "\n".join(rows)
