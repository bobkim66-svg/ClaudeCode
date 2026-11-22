"""gemini_export: Markdown 변환(순수 로직), 탭 고르기, 실제 Chrome(헤드리스)으로 가짜 보고서 저장."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from council.chrome import ChromeError, find_chrome
from council.export_md import TabInfo, first_heading, pick_tab, tree_to_markdown

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("fake_report_page.html").resolve().as_uri()

try:
    find_chrome()
    HAVE = importlib.util.find_spec("playwright") is not None
except ChromeError:
    HAVE = False
needs_chrome = pytest.mark.skipif(not HAVE, reason="Chrome 또는 Python Playwright 없음")


def e(t, *c, **k):
    return {"t": t, "c": list(c), **k}


def x(s):
    return {"t": "#text", "x": s}


# ---------- 변환 규칙 ----------

def test_headings_inline_and_links():
    md = tree_to_markdown(e("div", e("h2", x(" 제목 ")),
                            e("p", x("가 "), e("strong", x(" 굵게 ")), x("와 "), e("em", x("기울임")),
                              x(" "), e("a", x("링크"), h="https://a.org/x y)"), e("sup", x("3")))))
    assert md == "## 제목\n\n가 **굵게** 와 *기울임* [링크](https://a.org/x%20y%29)[3]\n"


def test_nested_and_ordered_lists():
    md = tree_to_markdown(e("div", e("ul", e("li", x("가"), e("ul", e("li", x("가-1")))), e("li", x("나"))),
                            e("ol", e("li", x("하나")), e("li", x("둘")))))
    assert md == "- 가\n  - 가-1\n- 나\n\n1. 하나\n2. 둘\n"


def test_table_escapes_pipe_and_keeps_line_breaks():
    md = tree_to_markdown(e("table", e("thead", e("tr", e("th", x("a")), e("th", x("b")))),
                            e("tbody", e("tr", e("td", x("1|2")), e("td", x("위"), e("br"), x("아래"))),
                              e("tr", e("td", x("짧은 행"))))))
    assert md == "| a | b |\n|---|---|\n| 1\\|2 | 위<br>아래 |\n| 짧은 행 | |\n"


def test_code_block_keeps_whitespace_and_quote():
    md = tree_to_markdown(e("div", e("pre", e("code", x("a = 1\n    b = 2\n"))), e("blockquote", e("p", x("인용")))))
    assert md == "```\na = 1\n    b = 2\n```\n\n> 인용\n"


def test_unknown_elements_and_whitespace():
    md = tree_to_markdown(e("custom-panel", x("\n   "), e("message-content", e("p", x("  문단   하나 "))),
                            e("span", x("인라인")), e("a", x("상대 링크"), h="javascript:void(0)")))
    assert md == "문단 하나\n\n인라인상대 링크\n"
    assert tree_to_markdown(None) == "" and tree_to_markdown(e("div", x("  "))) == ""
    assert first_heading("본문\n## 두번째\n# 첫") == "두번째" and first_heading("제목 없음") is None


# ---------- 탭 고르기 ----------

def _t(i, url="https://gemini.google.com/app/a", rank=None):
    return TabInfo(i, url, f"t{i}", rank)


def test_pick_tab_rules():
    G = "gemini.google.com"
    tabs = [_t(0, "https://www.google.com", rank=0), _t(1, rank=2), _t(2, rank=1)]
    chosen, cands = pick_tab(tabs, G)
    assert chosen.index == 2 and [c.index for c in cands] == [2, 1]   # Gemini 탭 중 가장 최근에 띄운 탭
    assert pick_tab([_t(0), _t(1, rank=5)], G)[0].index == 1          # 순서를 아는 탭 우선
    amb, cands = pick_tab([_t(0), _t(1)], G)
    assert amb is None and len(cands) == 2                             # 순서를 모름 → 번호로 묻기
    assert pick_tab([_t(0)], G)[0].index == 0                          # 후보가 하나뿐
    assert pick_tab([_t(0, "about:blank", rank=0)], G) == (None, [])


# ---------- 실제 Chrome + 가짜 보고서 ----------
# 탭은 Chrome 자체 기능(/json/new, /json/activate)으로 만들고 앞으로 띄운다 = 사용자가 탭을 클릭한 것과 같음.

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Chrome:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}/json"

    def _req(self, path, method="GET"):
        with urllib.request.urlopen(urllib.request.Request(self.base + path, method=method), timeout=10) as r:
            return r.read().decode("utf-8")

    def open(self, url):
        tid = json.loads(self._req("/new?" + urllib.parse.quote(url, safe=""), "PUT"))["id"]
        time.sleep(0.5)
        return tid

    def activate(self, tid):
        self._req("/activate/" + tid)
        time.sleep(0.3)


@pytest.fixture
def chrome(tmp_path):
    from council.chrome import launch_chrome
    port = _free_port()
    extra = ("--headless=new",) + (("--no-sandbox",) if hasattr(os, "geteuid") and os.geteuid() == 0 else ())
    proc = launch_chrome(port, tmp_path / "profile", start_url="about:blank", extra_args=extra)
    yield port, _Chrome(port)
    proc.kill()
    proc.wait(timeout=10)


def _export(tmp_path, port, *more, out="out", answer=""):
    """별도 프로세스로 실행 (exe 와 같은 경로). answer: 질문에 줄 표준입력."""
    args = [sys.executable, str(ROOT / "run_export.py"), "--once", "--port", str(port), "--out-dir",
            str(tmp_path / out), "--url-contains", "fake_report_page", "--profile-dir", str(tmp_path / "profile")]
    return subprocess.run(args + list(more), input=answer, capture_output=True, text=True, encoding="utf-8",
                          timeout=60, cwd=tmp_path)


def _md(tmp_path, out="out"):
    files = list((tmp_path / out).glob("*.md"))
    assert len(files) == 1, files
    return files[0], files[0].read_text(encoding="utf-8")


@needs_chrome
def test_export_front_tab_to_markdown(tmp_path, chrome):
    port, c = chrome
    b = c.open(PAGE)
    c.open(PAGE + "?title=다른 보고서")
    c.activate(b)                                              # 화면에 띄운 탭 = b
    r = _export(tmp_path, port)
    assert r.returncode == 0, r.stdout + r.stderr
    f, md = _md(tmp_path)
    assert "IR_기반_로컬_플래닝" in f.name and "다른 보고서" not in md
    assert md.startswith("> 원본 채팅: " + PAGE) and "상태: 완료" in md and "보고서 패널" in md
    assert "\n# IR 기반 로컬 플래닝 전환·복구\n" in md
    assert "**동시에 열화**" in md and "*열화상(LWIR)*" in md and "[1](https://example.org/src1)" in md
    assert "[원 논문](https://example.org/paper?id=2)" in md
    assert "| Kim 2024 | LWIR\\|LiDAR 융합 | 연기 **밀도** 높을 때<br>성능 저하 |" in md
    assert "- 로컬 플래너 수준의 센서 신뢰도 활용\n  - 전환 기준 미정립" in md and "1. 첫 단계\n2. 둘째 단계" in md
    assert "```\nreliability = w_ir * r_ir + w_rgb * r_rgb\n```" in md and "> 확인 필요" in md
    assert "1. [Thermal odometry survey](https://example.org/src1)" in md and md.endswith("가짜 보고서 끝 ✓\n")
    # 버튼·아이콘 이름·숨긴 글자·채팅 본문은 들어가지 않음
    assert not any(s in md for s in ("공유 및 내보내기", "more_vert", "숨겨진 글자", "연구 계획을 짜봤어요"))
    assert (tmp_path / "out" / "export_raw" / (f.stem + ".html")).exists()
    # 다른 탭을 앞으로 띄우면 그 탭을 저장
    c.activate(c.open(PAGE + "?title=세번째"))
    _export(tmp_path, port, out="out2")
    assert "\n# 세번째\n" in _md(tmp_path, "out2")[1]


@needs_chrome
def test_export_falls_back_to_last_answer_and_warns_when_generating(tmp_path, chrome):
    port, c = chrome
    c.activate(c.open(PAGE + "?panel=0"))
    r = _export(tmp_path, port)
    _, md = _md(tmp_path)
    assert r.returncode == 0 and "조사를 완료했습니다" in md and "마지막 답변" in md
    c.activate(c.open(PAGE + "?gen=1"))
    r2 = _export(tmp_path, port, out="out2", answer="n\n")
    assert "생성 중" in r2.stdout and not (tmp_path / "out2").exists()   # 'n' → 저장 안 함
    r3 = _export(tmp_path, port, "--yes", out="out3")
    assert r3.returncode == 0 and "생성 중(일부)" in _md(tmp_path, "out3")[1]


@needs_chrome
def test_export_no_gemini_tab_is_clear_message(tmp_path, chrome):
    port, _ = chrome
    r = _export(tmp_path, port)
    assert r.returncode == 2 and "Gemini 탭이 없습니다" in r.stdout
