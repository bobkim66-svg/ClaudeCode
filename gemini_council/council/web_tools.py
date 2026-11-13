"""자동 모드 준비 도구: 최초 로그인(--web-login)과 화면 구조 진단(--web-probe)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .backends.base import BackendError
from .backends.gemini_web import GeminiWebBackend, WebConfig
from .chrome import launch_chrome
from .roles import Role

PROBE_ROLE = Role("probe", "진단", "-")
PROBE_PROMPT = "연결 시험입니다. 'OK' 두 글자만 답하세요."

# 페이지에서 셀렉터 보정에 필요한 정보만 모은다 (대화 목록 등 전체 HTML 은 저장하지 않음)
_PROBE_JS = r"""
() => {
  const cut = (s, n) => (s || "").slice(0, n);
  const btns = [...document.querySelectorAll("button")].slice(0, 300).map(b => ({
    aria: b.getAttribute("aria-label"), cls: cut(b.className, 120), text: cut(b.innerText, 40),
    visible: !!(b.offsetWidth || b.offsetHeight)}));
  const editables = [...document.querySelectorAll("[contenteditable='true'], textarea")].map(e => ({
    tag: e.tagName.toLowerCase(), cls: cut(e.className, 120), role: e.getAttribute("role"),
    aria: e.getAttribute("aria-label"), parents: (() => { const p = []; let x = e.parentElement;
      for (let i = 0; i < 4 && x; i++, x = x.parentElement) p.push(x.tagName.toLowerCase() + (x.className ? "." + cut(String(x.className), 60).replace(/ /g, ".") : "")); return p; })()}));
  const custom = {};
  for (const el of document.querySelectorAll("*")) { const t = el.tagName.toLowerCase(); if (t.includes("-")) custom[t] = (custom[t] || 0) + 1; }
  const lastOf = sel => { const a = document.querySelectorAll(sel); return a.length ? cut(a[a.length - 1].outerHTML, 20000) : null; };
  const ed = document.querySelector("[contenteditable='true'], textarea");
  let composer = ed; for (let i = 0; i < 4 && composer && composer.parentElement; i++) composer = composer.parentElement;
  return { url: location.href, buttons: btns, editables, custom_elements: custom,
           last_model_response: lastOf("model-response"), last_message_content: lastOf("message-content"),
           composer_html: composer ? cut(composer.outerHTML, 20000) : null };
}
"""


def web_login(cfg: WebConfig, read: Callable[[str], str] = input, out: Callable[[str], None] = print) -> int:
    """전용 프로필 Chrome 을 Gemini 페이지로 띄우고, 사용자가 로그인한 뒤 Enter 를 누르면 입력창이 보이는지 확인."""
    launch_chrome(cfg.port, cfg.profile_dir, start_url=cfg.new_chat_url, extra_args=cfg.extra_args)
    out("자동 모드 전용 Chrome 창을 열었습니다 (평소 Chrome 과 별개의 프로필).")
    out("  1) 그 창에서 경북대 계정으로 Gemini 에 로그인하세요.")
    out("  2) 비평가용으로 쓸 모델(Pro 등)을 미리 골라 두세요 (프로그램은 모델을 바꾸지 않습니다).")
    read("로그인이 끝났으면 Enter: ")
    be = GeminiWebBackend(cfg, log=out)
    try:
        ctx = be.connect()
        page = next((p for p in ctx.pages if cfg.new_chat_url.split("?")[0] in p.url), None) or ctx.new_page()
        if cfg.new_chat_url.split("?")[0] not in page.url:
            page.goto(cfg.new_chat_url, timeout=cfg.page_load_timeout_s * 1000)
        be._pages = {PROBE_ROLE.key: page}
        be._tabs = {PROBE_ROLE.key: 1}
        be._wait_ready(page, PROBE_ROLE)
    except BackendError as e:
        out(f"확인 실패: {e}")
        return 1
    finally:
        be.close()
    out(f"확인 완료: 로그인 상태가 전용 프로필({cfg.profile_dir})에 저장되었습니다.")
    return 0


def web_probe(cfg: WebConfig, out_dir: Path, out: Callable[[str], None] = print) -> int:
    """새 채팅에서 시험 메시지 1개를 보내고, 셀렉터 보정용 정보를 out_dir 에 저장한다."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = out_dir / f"probe_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    report: dict = {"time": stamp, "steps": {}}
    be = GeminiWebBackend(cfg, log=out)
    code = 1
    try:
        ctx = be.connect()
        page = ctx.new_page()
        page.goto(cfg.new_chat_url, timeout=cfg.page_load_timeout_s * 1000)
        be._pages, be._tabs = {PROBE_ROLE.key: page}, {PROBE_ROLE.key: 1}
        be._reset_done.add(PROBE_ROLE.key)  # 진단 메시지에는 리셋 문구를 붙이지 않음
        try:
            be._wait_ready(page, PROBE_ROLE)
            report["steps"]["input_found"] = True
        except BackendError as e:
            report["steps"]["input_found"] = f"실패: {e}"
        for key in ("input", "send", "stop"):
            hit = next((css for css in be.sel[key] if page.locator(css).count() > 0), None)
            report["steps"][f"selector_{key}"] = hit
        page.screenshot(path=str(folder / "1_before.png"))
        if report["steps"]["input_found"] is True:
            try:
                reply = be.send(PROBE_ROLE, PROBE_PROMPT)
                report["steps"]["reply"] = reply[:500]
                code = 0 if reply.strip() else 1
            except BackendError as e:
                report["steps"]["reply"] = f"실패: {type(e).__name__}: {e}"
        report["page"] = page.evaluate(_PROBE_JS)
        page.screenshot(path=str(folder / "2_after.png"))
    except Exception as e:  # 진단 도구: 어떤 오류든 보고서에 남기고 사용자에게 보여 준 뒤 실패로 끝낸다
        report["error"] = f"{type(e).__name__}: {e}"
        out(f"진단 중 오류: {report['error']}")
    finally:
        be.close()
        (folder / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "성공" if code == 0 else "실패 (보고서를 보내 주세요)"
    out(f"진단 {verdict}: 답변 = {str(report['steps'].get('reply'))[:80]!r}")
    out(f"진단 결과 폴더: {folder}  (probe_report.json + 화면 캡처 2장)")
    out("※ 화면 캡처에는 Gemini 화면(왼쪽 대화 목록 포함)이 그대로 찍힙니다. 공유 전에 확인하세요.")
    return code
