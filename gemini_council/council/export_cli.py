"""gemini_export.exe: 전용 Chrome 창에 지금 띄워 둔 Gemini 채팅의 Deep Research 보고서를 .md 로 저장.

흐름: 전용 Chrome 에 연결(꺼져 있으면 띄움) → 화면에 보이는 Gemini 탭 고르기 → 보고서 영역을 Markdown 으로 변환
→ research/날짜_제목.md 저장 (+ 진단용 원본 HTML: research/export_raw/) → Enter 로 다른 채팅도 저장, q 로 종료.
읽기만 한다: 클릭·입력·전송을 하지 않으므로 사용량을 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backends.gemini_web import DEFAULT_SELECTORS, GEMINI_NEW_CHAT_URL
from .chrome import DEFAULT_PORT, ChromeError, cdp_endpoint, default_profile_dir, launch_chrome
from .export_md import activation_order, element_markdown, first_heading, pick_tab, report_root, tab_infos
from .research import DEFAULT_RESEARCH_SELECTORS, slugify
from .session import write_atomic


@dataclass
class ExportConfig:
    out_dir: Path = field(default_factory=lambda: Path("research"))
    port: int = DEFAULT_PORT
    profile_dir: Path = field(default_factory=default_profile_dir)
    start_url: str = GEMINI_NEW_CHAT_URL          # 전용 Chrome 을 새로 띄울 때 열 주소
    url_contains: str = "gemini.google.com"       # Gemini 탭으로 볼 주소 조건
    extra_args: tuple[str, ...] = ()
    save_raw: bool = True                         # 원본 HTML 도 저장 (변환 규칙 보정용)


LOAD_WAIT_S = 20.0


def _stamp(now: datetime) -> str:
    off = now.utcoffset()
    tz = "KST" if off is not None and off.total_seconds() == 9 * 3600 else f"UTC{now:%z}"
    return f"{now:%Y-%m-%d %H:%M:%S} ({tz})"


def export_page(page: Any, cfg: ExportConfig, now: datetime, ask: Callable[[str], str]) -> Path | None:
    """탭 하나의 보고서를 저장하고 경로를 돌려준다. 저장하지 않으면 None."""
    try:
        page.wait_for_load_state("load", timeout=LOAD_WAIT_S * 1000)  # 방금 연 창이면 아직 읽는 중일 수 있음
    except Exception:  # noqa: BLE001 — 시간 초과여도 보이는 만큼 읽는다
        pass
    root, where = report_root(page, DEFAULT_RESEARCH_SELECTORS["report"], DEFAULT_SELECTORS["response_container"])
    if root is None:
        print("  이 탭에서 보고서나 답변을 찾지 못했습니다. Deep Research 보고서가 열린 채팅인지 확인하세요.")
        return None
    generating = any(page.locator(css).first.is_visible() for css in DEFAULT_SELECTORS["stop"]
                     if page.locator(css).count())
    if generating and ask("  아직 생성 중입니다 (중지 버튼이 보임). 그래도 지금까지를 저장할까요? [y/N]: ").lower() != "y":
        return None
    md, html = element_markdown(root)
    if not md.strip():
        print(f"  {where} 은 찾았지만 내용이 비어 있습니다.")
        return None
    title = first_heading(md) or page.title() or "Gemini 보고서"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{now:%Y%m%d_%H%M%S}_{slugify(title)}"
    path = cfg.out_dir / f"{base}.md"
    k = 2
    while path.exists():  # 같은 초에 두 번 저장해도 덮어쓰지 않음
        path, k = cfg.out_dir / f"{base}_{k}.md", k + 1
    head = [f"> 원본 채팅: {page.url}  ",
            f"> 저장: {_stamp(now)} · 읽은 곳: {where} · 상태: {'생성 중(일부)' if generating else '완료'}", ""]
    if not md.lstrip().startswith("# "):
        head = [f"# {title}", ""] + head
    write_atomic(path, "\n".join(head) + "\n" + md)
    if cfg.save_raw:
        raw = cfg.out_dir / "export_raw" / f"{path.stem}.html"
        raw.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(raw, html)
    print(f"  저장: {path.resolve()}  ({len(md):,}자, {where})")
    return path


def choose_page(ctx: Any, cfg: ExportConfig, ask: Callable[[str], str]) -> Any | None:
    pages = list(ctx.pages)
    chosen, cands = pick_tab(tab_infos(pages, activation_order(cdp_endpoint(cfg.port))), cfg.url_contains)
    if chosen is not None:
        print(f"  대상 탭: {chosen.title or '(제목 없음)'}  {chosen.url}")
        return pages[chosen.index]
    if not cands:
        print("  전용 Chrome 창에 Gemini 탭이 없습니다. 그 창에서 저장할 채팅을 여세요.")
        return None
    print("  화면에 띄운 탭을 알 수 없습니다. 번호를 고르세요:")
    for k, t in enumerate(cands, 1):
        print(f"   {k}) {t.title or '(제목 없음)'}  {t.url}")
    raw = ask("  번호 [Enter = 취소]: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(cands):
        return pages[cands[int(raw) - 1].index]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="전용 Chrome 에 띄운 Gemini Deep Research 보고서를 .md 로 저장")
    p.add_argument("--out-dir", type=Path, default=Path("research"), help="저장 폴더")
    p.add_argument("--once", action="store_true", help="지금 화면의 탭 하나만 저장하고 끝냄")
    p.add_argument("--yes", action="store_true", help="생성 중이어도 묻지 않고 저장")
    p.add_argument("--no-raw", action="store_true", help="진단용 원본 HTML 을 저장하지 않음")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Chrome 원격 디버깅 포트")
    p.add_argument("--profile-dir", type=Path, default=None, help="전용 Chrome 프로필 폴더")
    p.add_argument("--web-url", default=None, help="(시험용) Chrome 을 새로 띄울 때 열 주소")
    p.add_argument("--url-contains", default="gemini.google.com", help="(시험용) Gemini 탭 주소 조건")
    p.add_argument("--chrome-arg", action="append", default=[], help="(시험용) Chrome 추가 옵션")
    return p


def run(argv: list[str], ask: Callable[[str], str] = input) -> int:
    args = build_parser().parse_args(argv)
    cfg = ExportConfig(out_dir=args.out_dir, port=args.port, extra_args=tuple(args.chrome_arg),
                       url_contains=args.url_contains, save_raw=not args.no_raw)
    if args.profile_dir is not None:
        cfg.profile_dir = args.profile_dir
    if args.web_url:
        cfg.start_url = args.web_url
    raw_ask = ask

    def ask(q: str) -> str:  # 입력이 끊기면(EOF) 빈 답 = 기본값(저장 안 함·취소·종료)
        if args.yes and "[y/N]" in q:
            return "y"
        try:
            return raw_ask(q)
        except EOFError:
            return "q" if q.lstrip().startswith("저장할 채팅을") else ""
    from playwright.sync_api import sync_playwright  # 무거운 모듈
    try:
        launched = launch_chrome(cfg.port, cfg.profile_dir, start_url=cfg.start_url, extra_args=cfg.extra_args)
    except ChromeError as e:
        print(f"시작할 수 없음: {e}")
        return 1
    if launched is not None:
        print("전용 Chrome 창을 새로 열었습니다. 그 창에서 저장할 채팅을 여세요.")
    saved = 0
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(cdp_endpoint(cfg.port))
        ctx = browser.contexts[0]
        while True:
            if not args.once:
                cmd = ask("\n저장할 채팅을 전용 창 화면에 띄운 뒤 Enter (q + Enter: 종료): ").strip().lower()
                if cmd == "q":
                    break
            page = choose_page(ctx, cfg, ask)
            if page is not None and export_page(page, cfg, datetime.now().astimezone(), ask) is not None:
                saved += 1
            if args.once:
                break
    # Chrome 창은 닫지 않는다 (사용자가 쓰던 창)
    print(f"저장한 보고서: {saved}개. 폴더: {cfg.out_dir.resolve()}")
    return 0 if saved or not args.once else 2


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    interactive = argv is None and len(sys.argv) == 1 and sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        return run(sys.argv[1:] if argv is None else argv)
    try:
        print("=" * 64)
        print(" Gemini 보고서 → Markdown 저장 (전용 Chrome 창에 띄운 채팅)")
        print("=" * 64)
        return run([])
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        print("\nEnter 를 누르면 창이 닫힙니다.", flush=True)
        try:
            input()
        except EOFError:
            pass
