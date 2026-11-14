"""Deep Research 사용 한도: 시도 기록(research_quota_log.json)과 다음 시도 시각 계산 (순수 로직).

확인된 사실 (2026-11-13 실제 화면): 한도 안내 문구
  "일일 조사 요청 1개를 모두 사용하셨습니다. 새로운 연구 요청은 2026. 11. 13. 이후에 진행하실 수 있습니다."
- 계획은 생성됐고, 안내는 '연구 시작'을 누른 뒤에 나왔다.
- 날짜만 나오고 시각은 없다. 11/13(한국)에 본 문구가 '11/13 이후' → 날짜가 미국 태평양 시간(PT)
  기준이라는 가설과 맞는다 (확인 필요).

확인되지 않은 것: 24시간 롤링인지 날짜 단위인지, 계획 생성만으로 차감되는지, 주간 한도 유무.
→ 규칙을 하나로 단정하지 않고 두 후보 중 늦은 쪽 + 여유 시간으로 다음 시도를 잡고, 다시 걸리면
  간격을 늘린다. 모든 시도를 기록해 실제 규칙을 데이터로 판정할 수 있게 한다.
시간은 모두 시간대 정보가 있는(aware) datetime 으로 다룬다. Windows exe 에는 zoneinfo 데이터가
없으므로 PT 는 미국 서머타임 규칙을 직접 계산한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .session import write_atomic

LIMIT_DATE_RE = re.compile(r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*이후")
WEEKLY_WORDS = ("주간", "이번 주", "weekly", "this week")


def parse_limit_date(message: str) -> date | None:
    """'새로운 연구 요청은 2026. 11. 13. 이후에' → date(2026, 11, 13). 없으면 None."""
    m = LIMIT_DATE_RE.search(message)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    first = d + timedelta(days=(6 - d.weekday()) % 7)
    return first + timedelta(weeks=n - 1)


def pacific_offset(at_utc: datetime) -> timedelta:
    """미국 태평양 시간의 UTC 차이. 서머타임: 3월 둘째 일요일 02:00 ~ 11월 첫째 일요일 02:00 (현지)."""
    y = at_utc.year
    start = datetime.combine(_nth_sunday(y, 3, 2), datetime.min.time(), timezone.utc) + timedelta(hours=10)
    end = datetime.combine(_nth_sunday(y, 11, 1), datetime.min.time(), timezone.utc) + timedelta(hours=9)
    return timedelta(hours=-7) if start <= at_utc.astimezone(timezone.utc) < end else timedelta(hours=-8)


def pacific_midnight(d: date) -> datetime:
    """PT 기준 d 날짜 00:00 을 UTC aware datetime 으로."""
    approx = datetime.combine(d, datetime.min.time(), timezone.utc) + timedelta(hours=8)
    return datetime.combine(d, datetime.min.time(), timezone.utc) - pacific_offset(approx)


BACKOFF_H = (1, 3, 6, 12)  # 같은 한도에 연속으로 걸릴 때 최소 간격 [시간], 이후는 마지막 값 반복


def next_try_time(now: datetime, message: str, last_start: datetime | None, attempt: int,
                  margin: timedelta = timedelta(minutes=30)) -> tuple[datetime, list[str]]:
    """한도에 걸린 뒤 다음 시도 시각과 그 근거.

    후보 (늦은 쪽 사용):
      - 지금 + 최소 간격 (연속 실패 횟수 attempt 에 따라 1→3→6→12 시간)
      - 안내 문구의 날짜 (PT 자정 가설) + 여유
      - 마지막 '연구 시작' + 24시간 (롤링 가설) + 여유
      - 주간 한도 문구면 지금 + 24시간
    """
    reasons: list[str] = []
    step = timedelta(hours=BACKOFF_H[min(attempt, len(BACKOFF_H) - 1)])
    candidates = [(now + step, f"최소 간격 {step.total_seconds() / 3600:.0f}시간")]
    d = parse_limit_date(message)
    if d is not None:
        candidates.append((pacific_midnight(d) + margin, f"안내 날짜 {d} (PT 자정 가설) + {margin.seconds // 60}분"))
    if last_start is not None:
        candidates.append((last_start + timedelta(hours=24) + margin,
                           f"마지막 연구 시작 + 24시간 (롤링 가설) + {margin.seconds // 60}분"))
    if any(w in message for w in WEEKLY_WORDS):
        candidates.append((now + timedelta(hours=24), "주간 한도 문구 → 24시간"))
    when, why = max(candidates, key=lambda c: c[0])
    reasons = [f"{'★' if c[1] == why else '-'} {c[1]}: {c[0].astimezone():%m-%d %H:%M}" for c in candidates]
    return when, reasons


# ------------------------------------------------------------------ 시도 기록

@dataclass
class QuotaLog:
    """research_quota_log.json = {"state": {...}, "events": [...]}

    state: last_start(실제로 연구가 진행된 마지막 '연구 시작' 시각), next_try(다음 시도 가능 예상),
           limit_streak(연속 한도 횟수), pending_start(방금 누른 '연구 시작' — 한도 안내가 나오면 버림)
    한도 안내로 끝난 클릭은 사용량이 차감되지 않았다고 보고 24시간 기준에서 뺀다 (가정, 기록으로 검증).
    events: 모든 시도의 시각·종류·문구 (실제 한도 규칙을 판정하기 위한 원자료)
    """

    path: Path
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "QuotaLog":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            return cls(path, data.get("state", {}), data.get("events", []))
        return cls(path)

    def save(self) -> None:
        write_atomic(self.path, json.dumps({"state": self.state, "events": self.events}, ensure_ascii=False, indent=2))

    def add(self, now: datetime, event: str, topic: str = "", message: str = "", **extra: Any) -> None:
        """event: sent / plan_shown / start_clicked / done / limit / failed"""
        self.events.append({"time": now.isoformat(timespec="seconds"), "event": event, "topic": topic,
                            **({"message": message[:500]} if message else {}), **extra})
        if event == "start_clicked":
            self.state["pending_start"] = now.isoformat(timespec="seconds")
        if event in ("done", "failed") and "pending_start" in self.state:  # 연구가 실제로 진행됨
            self.state["last_start"] = self.state.pop("pending_start")
        if event == "limit":
            self.state.pop("pending_start", None)
        if event == "done":
            self.state["limit_streak"] = 0
            self.state.pop("next_try", None)
        self.save()

    def _dt(self, key: str) -> datetime | None:
        v = self.state.get(key)
        return datetime.fromisoformat(v) if v else None

    @property
    def last_start(self) -> datetime | None:
        return self._dt("last_start")

    @property
    def next_try(self) -> datetime | None:
        return self._dt("next_try")

    def on_limit(self, now: datetime, topic: str, message: str) -> tuple[datetime, list[str]]:
        streak = int(self.state.get("limit_streak", 0))
        when, reasons = next_try_time(now, message, self.last_start, streak)
        self.state.update(limit_streak=streak + 1, next_try=when.isoformat(timespec="seconds"))
        self.add(now, "limit", topic, message, next_try=self.state["next_try"])
        return when, reasons
