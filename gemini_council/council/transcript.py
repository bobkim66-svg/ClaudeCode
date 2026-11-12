"""회의록(.md) 렌더링과 저장."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import CouncilConfig
from .protocol import Utterance


def render_markdown(
    topic: str,
    utterances: list[Utterance],
    cfg: CouncilConfig,
    stop_reason: str,
    started: datetime,
    finished: datetime,
    note: str = "",
    appendix: str = "",
) -> str:
    """회의록 마크다운 문자열 생성. 권장 길이를 넘은 발언은 표시만 하고 내용은 자르지 않는다."""
    rounds = max((u.round for u in utterances), default=0)
    over = [u for u in utterances if u.chars > cfg.char_limit]
    lines = [
        "# 토론 회의록",
        "",
        f"- 주제: {topic}",
        f"- 시작: {started:%Y-%m-%d %H:%M:%S} / 종료: {finished:%Y-%m-%d %H:%M:%S}",
        f"- 종료 사유: {stop_reason}",
        f"- 진행 라운드: {rounds} / 최대 {cfg.max_rounds}, 발언 수: {len(utterances)}",
        f"- 권장 길이({cfg.char_limit}자) 초과 발언: {len(over)}개",
    ]
    if note:
        lines.append(f"- 비고: {note}")
    current = 0
    for u in utterances:
        if u.round != current:
            current = u.round
            lines += ["", f"## 라운드 {current}"]
        flag = " ⚠ 길이 초과" if u.chars > cfg.char_limit else ""
        lines += ["", f"### {u.speaker_name} ({u.chars}자){flag}", "", u.text.strip()]
    if appendix:
        lines += ["", "## 사용량", "", appendix]
    return "\n".join(lines) + "\n"


def save_transcript(markdown: str, log_dir: Path, now: datetime) -> Path:
    """log_dir/discussion_YYYYMMDD_HHMMSS.md 로 저장. 같은 이름이 있으면 _1, _2 … 를 붙여 덮어쓰지 않는다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"discussion_{now:%Y%m%d_%H%M%S}"
    path = log_dir / f"{stem}.md"
    k = 1
    while path.exists():
        path = log_dir / f"{stem}_{k}.md"
        k += 1
    path.write_text(markdown, encoding="utf-8")
    return path
