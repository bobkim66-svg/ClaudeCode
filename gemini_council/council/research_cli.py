"""gemini_research.exe 진입점: Deep Research 대기열 실행, 진단(--probe / --probe-run), 최초 로그인."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from .backends.base import BackendError
from .backends.gemini_web import WebConfig
from .chrome import DEFAULT_PORT
from .console import ConsoleCommands
from .research import (MODES, ResearchConfig, ResearchRunner, Topic, TopicError, load_queue, parse_topics,
                       pending)
from .web_tools import _PROBE_JS, web_login

TOPICS_HELP = """# Deep Research 주제 목록: 한 줄에 하나, '모드 | 주제' (모드: 일반 / 사고, 생략하면 기본 모드)
# 예)
# 사고 | IR·LiDAR 융합 지각 신뢰도 기반 로컬 플래너 전환 선행연구
# 일반 | 열화상 카메라 깊이 추정 방법 비교
"""
_MODE_CANDIDATES_JS = r"""
(words) => {
  const out = [];
  const cut = (s, n) => (s || "").toString().slice(0, n);
  const all = [];                                   // 열린 shadow DOM 안까지 훑는다
  const walk = root => { for (const el of root.querySelectorAll("*")) { all.push(el); if (el.shadowRoot) walk(el.shadowRoot); } };
  walk(document);
  for (const el of all) {
    if (!(el.offsetWidth || el.offsetHeight)) continue;
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join("").trim();
    if (!own || own.length > 40 || !words.some(w => own.includes(w))) continue;
    const chain = []; let x = el;
    for (let i = 0; i < 6 && x; i++, x = x.parentElement)
      chain.push({tag: x.tagName.toLowerCase(), role: x.getAttribute("role"), aria: x.getAttribute("aria-label"),
                  testid: x.getAttribute("data-test-id"), popup: x.getAttribute("aria-haspopup"), cls: cut(x.className, 80)});
    out.push({text: own, chain});
    if (out.length >= 12) break;
  }
  return out;
}
"""
# 처음 기본값('OK' 어원 3문장)은 실제 실행에서 "언어 모델로서 생성할 수 없다" 답변을 받아 사용자 지정 주제로 교체
PROBE_TOPIC = "젠슨 황의 사퇴가 가져다줄 국제적 경제 시스템의 변화 및 대한민국의 경제 전략 방향성은 어떻게 달라지는가?"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemini Deep Research 대기열 (주제 여러 개를 차례로 자동 진행)")
    p.add_argument("--topics", type=Path, default=Path("research_topics.txt"), help="주제 목록 파일")
    p.add_argument("--mode", choices=MODES, default="사고", help="모드를 안 적은 주제의 기본 모드")
    p.add_argument("--interval", type=float, default=12.0, help="주제 사이 고정 간격 [분]")
    p.add_argument("--max-minutes", type=float, default=180.0, help="연구 1개 최대 대기 [분] (넘으면 부분 저장)")
    p.add_argument("--raw", action="store_true", help="선행연구 정리 요구(템플릿)를 붙이지 않고 주제만 보냄")
    p.add_argument("--out-dir", type=Path, default=Path("research"), help="보고서 저장 폴더")
    p.add_argument("--web-login", action="store_true", help="전용 Chrome 프로필로 Gemini 로그인 (최초 1회)")
    p.add_argument("--probe", action="store_true", help="화면 구조 진단: 버튼·메뉴 이름만 확인 (메시지 안 보냄)")
    p.add_argument("--probe-run", action="store_true", help="시험 주제로 Deep Research 1회 실제 진행 (사용량 1회)")
    p.add_argument("--probe-topic", default=PROBE_TOPIC, help="--probe-run 에 쓸 주제")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Chrome 원격 디버깅 포트")
    p.add_argument("--profile-dir", type=Path, default=None, help="전용 Chrome 프로필 폴더")
    p.add_argument("--web-url", default=None, help="(시험용) 새 채팅 주소 대신 열 주소")
    p.add_argument("--chrome-arg", action="append", default=[], help="(시험용) Chrome 추가 옵션")
    p.add_argument("--plan-minutes", type=float, default=0.0,
                   help="'연구 시작' 버튼을 기다리는 최대 시간 [분] (0 = 제한 없음, 기본)")
    p.add_argument("--stable-sec", type=float, default=120.0, help="(시험용) 보고서가 그대로여야 하는 시간 [s]")
    return p


def _log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def configs(args: argparse.Namespace) -> tuple[WebConfig, ResearchConfig]:
    web = WebConfig(port=args.port, extra_args=tuple(args.chrome_arg))
    if args.profile_dir is not None:
        web.profile_dir = args.profile_dir
    if args.web_url:
        web.new_chat_url = args.web_url
    rc = ResearchConfig(interval_s=args.interval * 60.0, research_timeout_s=args.max_minutes * 60.0,
                        stable_s=args.stable_sec, done_stable_s=min(15.0, args.stable_sec), out_dir=args.out_dir,
                        plan_timeout_s=args.plan_minutes * 60.0,
                        template=not args.raw)
    return web, rc


def research_probe(web: WebConfig, rc: ResearchConfig, run: bool, default_mode: str,
                   out: Callable[[str], None] = print, topic: str = PROBE_TOPIC) -> int:
    """메뉴를 열어 실제 이름을 기록한다. run=True 면 짧은 주제로 Deep Research 를 끝까지 1회 진행."""
    folder = rc.out_dir / f"probe_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    report: dict = {"steps": {}}
    r = ResearchRunner(web, rc, log=out)
    code = 1
    try:
        r.ctx = r.web.connect()
        page = r.ctx.new_page()
        page.goto(web.new_chat_url, timeout=web.page_load_timeout_s * 1000)
        r.web._tabs = {r.ROLE.key: 1}
        r.web._wait_ready(page, r.ROLE)
        page.bring_to_front()
        menu_names = "() => [...document.querySelectorAll('[role^=menuitem],[role=option]')]" \
                     ".filter(e => e.offsetWidth || e.offsetHeight).map(e => e.innerText.trim()).slice(0, 40)"
        # 모드 글자를 가진 요소의 구조 (선택 규칙을 정확히 맞추기 위함)
        import time as _t
        t0 = _t.monotonic()
        model_btn = r.find_model_button(page, wait_s=15.0)                   # 늦게 그려지는지 확인
        report["steps"]["model_found_after_s"] = round(_t.monotonic() - t0, 1) if model_btn else None
        report["steps"]["mode_candidates"] = page.evaluate(_MODE_CANDIDATES_JS, r.sel["mode_words"])
        report["steps"]["text_hits"] = {w: page.get_by_text(w).count() for w in r.sel["mode_words"]}
        report["steps"]["frames"] = [f.url[:80] for f in page.frames]
        try:  # 입력창 주변의 접근성 구조 (버튼 역할·이름). 대화 목록이 섞이지 않게 입력창 조상 일부만.
            box = r.web._first_visible(page, "input")
            area = page.locator("body")
            if box is not None:  # 실제로 있는 가장 높은 조상(최대 6단계)
                for lv in range(6, 0, -1):
                    cand = box.locator(f"xpath=ancestor::*[{lv}]")
                    if cand.count() > 0 and cand.first.evaluate("e => e.tagName") not in ("BODY", "HTML"):
                        area = cand.first
                        break
            report["steps"]["composer_aria"] = area.aria_snapshot(timeout=5000)[:6000]
        except Exception as e:  # 진단 정보 수집 실패는 보고서에 남김
            report["steps"]["composer_aria"] = f"수집 실패: {type(e).__name__}: {e}"
        page.screenshot(path=str(folder / "0_start.png"))
        report["steps"]["model_button"] = model_btn.inner_text() if model_btn else None
        report["steps"]["model_button_rule_hit"] = r._visible(page, r.sel["model_button"]) is not None
        if model_btn:
            model_btn.click()
            page.wait_for_timeout(800)
            report["steps"]["model_menu_items"] = page.evaluate(menu_names)
            page.screenshot(path=str(folder / "1_model_menu.png"))
            page.keyboard.press("Escape")
        tools = r._visible(page, r.sel["tools_button"])
        report["steps"]["tools_button"] = tools.get_attribute("aria-label") if tools else None
        if tools:
            tools.click()
            page.wait_for_timeout(800)
            report["steps"]["tools_menu_items"] = page.evaluate(menu_names)
            more = r._menu_item(page, r.sel["more_tools_texts"])
            if more:
                more.click()
                page.wait_for_timeout(800)
                report["steps"]["more_tools_items"] = page.evaluate(menu_names)
            page.screenshot(path=str(folder / "2_tools_menu.png"))
            page.keyboard.press("Escape")
        ok = bool(model_btn and tools)
        if run and ok:
            try:
                result = r.run_one(Topic(default_mode, topic))
                report["steps"]["run"] = result
                ok = result["status"] == "done"
            except BackendError as e:
                report["steps"]["run"] = f"실패: {type(e).__name__}: {e}"
                ok = False
            report["page"] = r.ctx.pages[-1].evaluate(_PROBE_JS)
            r.ctx.pages[-1].screenshot(path=str(folder / "3_after_run.png"))
        code = 0 if ok else 1
    except Exception as e:  # 진단 도구: 어떤 오류든 보고서에 남기고 실패로 끝낸다 (숨기지 않음)
        report["error"] = f"{type(e).__name__}: {e}"
        out(f"진단 중 오류: {report['error']}")
    finally:
        r.web.close()
        (folder / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    steps = report["steps"]
    out(f"모델 버튼: {steps.get('model_button')!r} / 모델 메뉴: {steps.get('model_menu_items')}")
    out(f"'+' 버튼: {steps.get('tools_button')!r} / 도구 메뉴: {steps.get('tools_menu_items')} "
        f"/ 도구 더보기: {steps.get('more_tools_items')}")
    out(f"진단 {'성공' if code == 0 else '실패 (아래 폴더의 probe_report.json 을 보내 주세요)'}: {folder}")
    out("※ 화면 캡처에는 Gemini 화면이 그대로 찍힙니다. 공유 전에 확인하세요.")
    return code


def ask_topics(path: Path, read: Callable[[str], str], default_mode: str) -> None:
    """주제 파일이 없으면 콘솔에서 받아 만든다."""
    print(f"주제 목록({path.name})이 없습니다. 한 줄에 하나씩 입력하세요. 빈 줄을 입력하면 끝납니다.")
    print(f"  형식: '모드 | 주제' (모드: 일반 / 사고, 생략하면 {default_mode})")
    lines = []
    while True:
        line = (read("  주제: ") or "").strip()
        if not line:
            break
        try:
            parse_topics(line, default_mode)
        except TopicError as e:
            print(f"    {e}")
            continue
        lines.append(line)
    path.write_text(TOPICS_HELP + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  저장: {path.resolve()} (메모장으로 고칠 수 있음)")


def run(argv: list[str], console: ConsoleCommands | None) -> int:
    args = build_parser().parse_args(argv)
    web, rc = configs(args)
    if args.web_login:
        return web_login(web)
    if not web.profile_dir.exists():
        print("처음 실행합니다. 먼저 전용 Chrome 에서 로그인합니다 (자동 토론 모드와 같은 프로필).")
        read = (lambda q: (print(q, end="", flush=True), console.wait_line())[1] or "") if console else input
        if web_login(web, read=read) != 0:
            return 1
    if args.probe or args.probe_run:
        return research_probe(web, rc, run=args.probe_run, default_mode=args.mode, topic=args.probe_topic)
    if not args.topics.exists():
        if console is None or not sys.stdin.isatty():
            print(f"주제 목록 파일이 없음: {args.topics}")
            return 1
        ask_topics(args.topics, lambda q: (print(q, end="", flush=True), console.wait_line())[1] or "", args.mode)
    try:
        topics = parse_topics(args.topics.read_text(encoding="utf-8-sig"), args.mode)  # 메모장 BOM 허용
    except TopicError as e:
        print(f"주제 목록 오류 ({args.topics}): {e}")
        return 1
    todo = pending(topics, load_queue(rc.queue_file))
    for t in todo:
        print(f"  - {t.mode} | {t.text}")
    if todo:
        hours = (len(todo) - 1) * args.interval / 60
        print(f"남은 주제 {len(todo)}개. 주제 사이 {args.interval:g}분 고정 간격 (간격만 약 {hours:.1f}시간 + 연구 시간). "
              "n + Enter: 대기 건너뛰기, q + Enter: 저장 후 종료")
    runner = ResearchRunner(web, rc, log=_log, commands=(console.poll if console else (lambda: None)))
    try:
        counts = runner.run_queue(topics)
    except BackendError as e:
        print(f"시작할 수 없음: {e}")
        return 1
    print(f"결과: 완료 {counts['done']}개, 실패 {counts['failed']}개, 남음 {counts['left']}개. 보고서 폴더: {rc.out_dir.resolve()}")
    return 0 if counts["left"] == 0 and counts["failed"] == 0 else 2


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    interactive = argv is None and len(sys.argv) == 1 and sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        return run(sys.argv[1:] if argv is None else argv, None if argv is not None else ConsoleCommands())
    console: ConsoleCommands | None = None
    try:
        print("=" * 64)
        print(" Gemini Deep Research 대기열 (주제를 차례로 자동 진행)")
        print("=" * 64)
        mode = ""
        while mode not in ("1", "2"):
            mode = input("모드를 안 적은 주제의 기본 모드 — 1: 일반, 2: 사고 [Enter = 2]: ").strip() or "2"
        raw = input("주제 사이 간격(분, 고정) [Enter = 12]: ").strip()
        argv2 = ["--mode", "일반" if mode == "1" else "사고"] + (["--interval", raw] if raw else [])
        console = ConsoleCommands()
        return run(argv2, console)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        print("\nEnter 를 누르면 창이 닫힙니다.", flush=True)
        if console is not None:
            console.wait_line()
        else:
            try:
                input()
            except EOFError:
                pass
