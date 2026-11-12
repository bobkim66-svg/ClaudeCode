"""진행 상태 저장/불러오기: 발언마다 자동 저장하고, 중단된 토론을 이어서 진행할 수 있게 한다.

파일 구성 (같은 폴더, 같은 이름):
  discussion_YYYYMMDD_HHMMSS.md   사람이 읽는 회의록
  discussion_YYYYMMDD_HHMMSS.json 프로그램이 읽는 진행 상태 (이어 하기용)
쓰기는 임시 파일에 쓴 뒤 교체하므로, 저장 도중 창이 닫혀도 파일이 반쯤 깨지지 않는다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .protocol import Utterance

STATE_VERSION = 1
STATUS_RUNNING = "RUNNING"
# 이어 하기 대상에서 빼는 상태 (정상 종료, 최대 라운드 도달은 '중단'이 아님)
FINISHED_STATUSES = {"COMPLETED", "MAX_ROUNDS"}


class SessionError(ValueError):
    """진행 상태 파일을 읽을 수 없거나 이어 할 수 없음."""


def write_atomic(path: Path, text: str) -> None:
    """임시 파일에 쓴 뒤 os.replace 로 교체 (Windows 에서도 원자적 교체)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def state_path_for(md_path: Path) -> Path:
    return md_path.with_suffix(".json")


def save_state(json_path: Path, data: dict[str, Any]) -> None:
    write_atomic(json_path, json.dumps(data, ensure_ascii=False, indent=2))


def load_state(path: Path) -> tuple[dict[str, Any], Path]:
    """회의록(.md) 또는 상태(.json) 경로를 받아 (상태, json 경로) 를 돌려준다."""
    p = Path(path)
    json_path = p if p.suffix.lower() == ".json" else state_path_for(p)
    if not json_path.exists():
        raise SessionError(f"진행 상태 파일이 없음: {json_path} (이 기능 이전에 만든 회의록은 이어 할 수 없음)")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SessionError(f"진행 상태 파일이 손상됨: {json_path} ({e})") from e
    if data.get("version") != STATE_VERSION:
        raise SessionError(f"지원하지 않는 상태 파일 버전: {data.get('version')}")
    for key in ("topic", "max_rounds", "utterances", "md_name", "status"):
        if key not in data:
            raise SessionError(f"상태 파일에 '{key}' 가 없음: {json_path}")
    return data, json_path


def utterances_from_state(data: dict[str, Any]) -> list[Utterance]:
    return [Utterance(int(u["round"]), u["speaker_key"], u["speaker_name"], u["text"]) for u in data["utterances"]]


def utterances_to_state(utts: list[Utterance]) -> list[dict[str, Any]]:
    return [{"round": u.round, "speaker_key": u.speaker_key, "speaker_name": u.speaker_name, "text": u.text}
            for u in utts]


def find_unfinished(log_dir: Path) -> Path | None:
    """log_dir 에서 가장 최근의 '중단된' 토론 상태 파일. 없으면 None. 깨진 파일은 건너뛴다."""
    if not log_dir.is_dir():
        return None
    for p in sorted(log_dir.glob("discussion_*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            data, _ = load_state(p)
        except SessionError:
            continue
        if data["status"] not in FINISHED_STATUSES:
            return p
    return None
