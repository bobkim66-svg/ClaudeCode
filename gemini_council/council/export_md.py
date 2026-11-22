"""화면에 띄운 Gemini 채팅의 Deep Research 보고서를 Markdown 으로 (순수 로직 + 페이지에서 읽기).

왜 화면 글자(inner_text)가 아니라 문서 구조를 읽는가: 글자만 긁으면 제목·표·링크·굵은 글씨가 사라진다.
페이지 안에서 보고서 영역의 DOM 을 작은 트리(JSON)로 바꿔 가져오고(_SERIALIZE_JS), 변환은 Python 에서 한다.
→ 변환 규칙은 Chrome 없이 단위 테스트할 수 있다.

트리 노드: {"t": 태그(소문자) 또는 "#text", "x": 글자(#text), "h": 링크 주소, "s": 이미지 주소, "a": alt, "c": [자식]}
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

# 페이지 안에서 실행: 보이지 않는 요소, 버튼·아이콘(글자 대신 'more_vert' 같은 아이콘 이름이 들어 있음)은 뺀다.
_SERIALIZE_JS = r"""root => {
  const SKIP = new Set(["SCRIPT","STYLE","NOSCRIPT","TEMPLATE","BUTTON","SVG","MAT-ICON","INPUT","TEXTAREA","SELECT"]);
  const ICON = /(material-icons|material-symbols|google-symbols)/;
  function walk(n) {
    if (n.nodeType === 3) return n.nodeValue ? {t: "#text", x: n.nodeValue} : null;
    if (n.nodeType !== 1) return null;
    const tag = n.tagName.toUpperCase();
    if (SKIP.has(tag) || n.getAttribute("role") === "button") return null;
    if (typeof n.className === "string" && ICON.test(n.className)) return null;
    const cs = getComputedStyle(n);
    if (cs.display === "none" || cs.visibility === "hidden") return null;
    const o = {t: tag.toLowerCase(), c: []};
    if (tag === "A" && n.href) o.h = n.href;
    if (tag === "IMG") { o.s = n.src; o.a = n.alt || ""; }
    const kids = n.shadowRoot ? n.shadowRoot.childNodes : n.childNodes;
    for (const k of kids) { const w = walk(k); if (w) o.c.push(w); }
    return o;
  }
  return walk(root);
}"""

BLOCK = {"p", "div", "section", "article", "header", "footer", "main", "aside", "nav", "figure", "figcaption",
         "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "pre", "blockquote", "hr", "dl", "dt", "dd",
         "details", "summary"}
HEADINGS = {f"h{i}": i for i in range(1, 7)}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


class _Renderer:
    """트리 → Markdown. 블록 요소는 앞뒤 빈 줄로 감싸고, 마지막에 빈 줄을 정리한다."""

    def is_block(self, n: dict) -> bool:
        if n.get("t") == "#text":
            return False
        return n["t"] in BLOCK or any(self.is_block(c) for c in n.get("c", []))

    def children(self, n: dict, pre: bool = False) -> str:
        return "".join(self.render(c, pre) for c in n.get("c", []))

    def render(self, n: dict, pre: bool = False) -> str:
        t = n.get("t")
        if t == "#text":
            return n.get("x", "") if pre else _norm(n.get("x", ""))
        if t in HEADINGS:
            text = _norm(self.children(n)).strip()
            return f"\n\n{'#' * HEADINGS[t]} {text}\n\n" if text else ""
        if t in ("p", "summary", "figcaption", "dt", "dd"):
            return f"\n\n{self.children(n).strip()}\n\n"
        if t == "br":
            return "\n" if pre else "  \n"
        if t == "hr":
            return "\n\n---\n\n"
        if t in ("strong", "b"):
            return self._wrap(self.children(n), "**")
        if t in ("em", "i"):
            return self._wrap(self.children(n), "*")
        if t in ("s", "del"):
            return self._wrap(self.children(n), "~~")
        if t == "code" and not pre:
            inner = self.children(n, pre=True)
            return f"`{inner}`" if "`" not in inner else f"`` {inner} ``"
        if t == "pre":
            code = self.children(n, pre=True).strip("\n")
            return f"\n\n```\n{code}\n```\n\n"
        if t == "a":
            text = _norm(self.children(n)).strip()
            href = n.get("h", "")
            if not href.startswith(("http://", "https://")):
                return text
            return f"[{text or href}]({href.replace(')', '%29').replace(' ', '%20')})"
        if t == "img":
            return f"![{n.get('a', '')}]({n.get('s', '')})" if n.get("s", "").startswith("http") else ""
        if t == "sup":
            text = _norm(self.children(n)).strip()
            return f"[{text}]" if text else ""  # 출처 번호: [1] 또는 [[1](주소)]
        if t in ("ul", "ol"):
            return self._list(n, ordered=(t == "ol"))
        if t == "li":  # 목록 밖의 li
            return f"\n\n- {self.children(n).strip()}\n\n"
        if t == "table":
            return self._table(n)
        if t == "blockquote":
            body = _cleanup(self.children(n))
            return "\n\n" + "\n".join(f"> {ln}" if ln else ">" for ln in body.split("\n")) + "\n\n"
        inner = self.children(n, pre)
        return f"\n\n{inner.strip()}\n\n" if t in BLOCK or self.is_block(n) else inner

    @staticmethod
    def _wrap(inner: str, mark: str) -> str:
        core = inner.strip()
        if not core:
            return inner
        lead = inner[: len(inner) - len(inner.lstrip())]
        trail = inner[len(inner.rstrip()):]
        return f"{lead}{mark}{core}{mark}{trail}"

    def _list(self, n: dict, ordered: bool) -> str:
        items = [c for c in n.get("c", []) if c.get("t") == "li"]
        out = []
        for k, li in enumerate(items, 1):
            marker = f"{k}. " if ordered else "- "
            body = re.sub(r"\n\n(?=(- |\d+\. ))", "\n", _cleanup(self.children(li)))  # 하위 목록은 붙여 씀
            lines = body.split("\n")
            pad = " " * len(marker)
            out.append(marker + lines[0] + "".join("\n" + (pad + ln if ln else "") for ln in lines[1:]))
        # 목록 항목 사이에 빈 줄이 있으면(항목 안에 문단) 그대로 두고, 아니면 붙여 쓴다
        return "\n\n" + "\n".join(out) + "\n\n"

    def _rows(self, n: dict) -> list[dict]:
        rows = []
        for c in n.get("c", []):
            if c.get("t") == "tr":
                rows.append(c)
            elif c.get("t") in ("thead", "tbody", "tfoot"):
                rows.extend(self._rows(c))
        return rows

    def _table(self, n: dict) -> str:
        rows = []
        for tr in self._rows(n):
            cells = [c for c in tr.get("c", []) if c.get("t") in ("td", "th")]
            rows.append([_escape_cell(_cleanup(self.children(c)).replace("\n\n", "<br>").replace("  \n", "<br>")
                                      .replace("\n", "<br>")) for c in cells])
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n\n" + "\n".join(lines) + "\n\n"


def _cleanup(md: str) -> str:
    """줄 끝 공백(강제 줄바꿈 '  ' 제외) 정리, 빈 줄 3개 이상 → 1개, 앞뒤 빈 줄 제거. 코드 블록 안은 건드리지 않는다."""
    out, in_code = [], False
    for ln in md.split("\n"):
        if ln.lstrip().startswith("```"):
            in_code = not in_code
            out.append(ln.rstrip())
            continue
        if in_code:
            out.append(ln)
            continue
        hard = ln.endswith("  ") and ln.strip()
        ln = ln.rstrip()
        if not ln.strip():
            ln = ""
        elif not re.match(r"^( {2,}|> )", ln):  # 목록 들여쓰기·인용은 유지
            ln = ln.lstrip()
        ln = re.sub(r"(?<=\S) {2,}(?=\S)", " ", ln)  # 인라인 요소 경계에서 생긴 겹친 공백
        out.append(ln + ("  " if hard else ""))
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def tree_to_markdown(tree: dict | None) -> str:
    """직렬화된 DOM 트리 → Markdown 문자열 (끝에 줄바꿈 하나)."""
    if not tree:
        return ""
    md = _cleanup(_Renderer().render(tree))
    return md + "\n" if md else ""


def first_heading(md: str) -> str | None:
    m = re.search(r"^#{1,3} (.+)$", md, flags=re.M)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------ 탭 고르기
#
# '화면에 띄운 탭' 판정: Chrome 의 /json/list 는 탭을 '최근에 앞으로 띄운 순서'로 돌려준다 (실측: 탭을 클릭해
# 활성화하면 맨 앞으로 옴, Chromium 153 / Xvfb 창 모드). Playwright 로 연결하면 모든 탭이 visibilityState
# 'visible'·hasFocus true 로 보여서 그 값으로는 고를 수 없었다 (실측). 순서를 못 읽으면 번호로 묻는다.

@dataclass
class TabInfo:
    index: int          # Playwright ctx.pages 안의 위치
    url: str
    title: str
    rank: int | None    # 최근 활성화 순서 (0 = 가장 최근에 앞으로 띄운 탭). 모르면 None


def pick_tab(tabs: list[TabInfo], url_contains: str) -> tuple[TabInfo | None, list[TabInfo]]:
    """(고른 탭, 후보 목록). 주소 조건에 맞는 탭 중 가장 최근에 앞으로 띄운 탭.
    순서를 모르고 후보가 여럿이면 (None, 후보) → 번호로 묻는다. 후보가 없으면 (None, [])."""
    cands = [t for t in tabs if url_contains in t.url]
    if not cands:
        return None, []
    ranked = sorted((t for t in cands if t.rank is not None), key=lambda t: t.rank)
    if ranked:
        return ranked[0], ranked + [t for t in cands if t.rank is None]
    if len(cands) == 1:
        return cands[0], cands
    return None, cands


# ------------------------------------------------------------------ 페이지에서 읽기 (Playwright)

def activation_order(endpoint: str, timeout_s: float = 3.0) -> list[str]:
    """Chrome /json/list 의 page 대상 id (최근 활성화 순). 실패하면 빈 목록."""
    try:
        with urllib.request.urlopen(endpoint + "/json/list", timeout=timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return []
    return [d["id"] for d in data if d.get("type") == "page" and "id" in d]


def tab_infos(pages: list[Any], order: list[str]) -> list[TabInfo]:
    rank = {tid: k for k, tid in enumerate(order)}
    out = []
    for i, p in enumerate(pages):
        try:
            cdp = p.context.new_cdp_session(p)
            tid = cdp.send("Target.getTargetInfo")["targetInfo"]["targetId"]
            cdp.detach()
            title = p.title()
        except Exception:  # noqa: BLE001 — 닫히는 중인 탭 등은 후보에서 뺀다
            continue
        out.append(TabInfo(i, p.url, title, rank.get(tid)))
    return out


def report_root(page: Any, report_selectors: list[str], response_selectors: list[str]) -> tuple[Any | None, str]:
    """(보고서 영역, 어디서 찾았는지). Deep Research 패널을 먼저, 없으면 마지막 답변."""
    best, best_len, where = None, 0, ""
    for css in report_selectors:
        loc = page.locator(css)
        for i in range(loc.count()):
            el = loc.nth(i)
            if el.is_visible():
                n = len(el.inner_text())
                if n > best_len:
                    best, best_len, where = el, n, f"보고서 패널 ({css})"
    if best is not None and best_len > 0:
        return best, where
    for css in response_selectors:
        loc = page.locator(css)
        if loc.count():
            return loc.last, f"마지막 답변 ({css})"
    return None, ""


def element_markdown(el: Any) -> tuple[str, str]:
    """(Markdown, 원본 HTML). 원본은 변환이 어긋났을 때 규칙을 고치기 위한 진단 자료."""
    tree = el.evaluate(_SERIALIZE_JS)
    return tree_to_markdown(tree), el.evaluate("e => e.outerHTML")
