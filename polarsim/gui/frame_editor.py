"""로봇 프레임 편집기: 막대(판) 목록을 표로 편집한다.

한 행 = 막대 하나. 같은 막대를 두 가지로 입력할 수 있고 서로 자동으로 맞춰진다:
  - 두 끝점 (x1, y1) → (x2, y2)
  - 끝점 1 + 각도 [°] + 길이   (각도·길이를 고치면 끝점 2 가 다시 계산된다)
방사면은 끝점 1 → 끝점 2 방향으로 걸을 때의 왼쪽/오른쪽. 미리보기의 주황 화살표가 실제 방사 방향이다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..config import ConfigError
from ..fdtd import EMIT_SIDES, Bar, FramePreset, frame_presets

MAX_BARS = 12
COLS = ["이름", "x1", "y1", "x2", "y2", "각도 [°]", "길이", "두께", "방사면", "진폭 ξ0 [m]", "위상 [°]"]
C_NAME, C_X1, C_Y1, C_X2, C_Y2, C_ANG, C_LEN, C_THK, C_EMIT, C_AMP, C_PH = range(len(COLS))
EMIT_LABELS = {"none": "없음 (벽)", "left": "왼쪽 면", "right": "오른쪽 면", "both": "양쪽 면"}
TOOLTIPS = {
    C_X1: "끝점 1 x [m]", C_Y1: "끝점 1 y [m]", C_X2: "끝점 2 x [m] (고치면 각도·길이 갱신)",
    C_Y2: "끝점 2 y [m] (고치면 각도·길이 갱신)", C_ANG: "끝점 1 에서 본 방향, x 축에서 반시계 [°] (고치면 끝점 2 갱신)",
    C_LEN: "중심선 길이 [m] (고치면 끝점 2 갱신)", C_THK: "막대 두께 [m] (중심선 양쪽으로 절반씩)",
    C_EMIT: "끝점 1 → 끝점 2 로 걸을 때의 왼쪽/오른쪽 면", C_AMP: "방사면 변위 진폭 [m]", C_PH: "방사면 위상 [°]",
}


def _fmt(v: float) -> str:
    return f"{v:.6g}"


class FrameEditor(QtWidgets.QWidget):
    """changed: 표 내용이 바뀔 때 (미리보기 갱신용). selectionChanged: 선택한 막대 행 (강조 표시용)."""

    changed = QtCore.Signal()
    selectionChanged = QtCore.Signal(int)
    presetLoaded = QtCore.Signal(object)  # FramePreset

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        for c, tip in TOOLTIPS.items():
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(190)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(lambda: self.selectionChanged.emit(self.current_row()))

        self.preset_box = QtWidgets.QComboBox()
        self._presets = frame_presets()
        for key, pr in self._presets.items():
            self.preset_box.addItem(pr.title, key)
        load = QtWidgets.QPushButton("예시 불러오기")
        load.clicked.connect(lambda: self.load_preset(self.preset_box.currentData()))

        def btn(text, slot, tip):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.preset_box, 1)
        row1.addWidget(load)
        row2 = QtWidgets.QHBoxLayout()
        for b in (btn("추가", self.add_bar, "새 막대 (선택한 막대 아래)"),
                  btn("복제", self.duplicate, "선택한 막대를 복사 (조금 옮겨서)"),
                  btn("삭제", self.remove, "선택한 막대 삭제"),
                  btn("▲ 위로", lambda: self.move(-1), "표에서 위로"), btn("▼ 아래로", lambda: self.move(1), "표에서 아래로"),
                  btn("끝점 뒤집기", self.flip, "끝점 1 ↔ 2 교환 (왼쪽/오른쪽 방사면도 함께 바꿔 같은 면 유지)")):
            row2.addWidget(b)
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(btn("파일로 저장…", self.save_dialog, "막대 목록을 JSON 파일로 저장"))
        row3.addWidget(btn("파일 불러오기…", self.load_dialog, "저장한 JSON 파일 불러오기"))
        row3.addStretch(1)
        hint = QtWidgets.QLabel(
            "<small>• 끝점 두 개를 넣거나, 끝점 1 + 각도 + 길이를 넣으세요 (서로 자동 계산).<br>"
            "• 방사면 왼쪽/오른쪽 = 끝점 1 → 끝점 2 방향으로 걸을 때 기준. 그림의 주황 화살표가 방사 방향입니다.<br>"
            "• 표에서 막대를 고르면 그림에 빨간 테두리로 강조됩니다. 단위: m, 각도·위상은 도(°).</small>")
        hint.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row1)
        lay.addWidget(self.table)
        lay.addLayout(row2)
        lay.addLayout(row3)
        lay.addWidget(hint)

    # ------------------------------------------------------------------ 행 읽기/쓰기
    def current_row(self) -> int:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return rows[0] if rows else -1

    def _set(self, r: int, c: int, text: str) -> None:
        item = self.table.item(r, c)
        if item is None:
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        else:
            item.setText(text)

    def _write_row(self, r: int, b: Bar) -> None:
        self._updating = True
        try:
            # 콤보박스를 먼저 만든다: 숫자 칸을 채우는 도중에 다른 신호가 이 행을 읽어도 형식이 완전하도록
            combo = self.table.cellWidget(r, C_EMIT)
            if combo is None:
                combo = QtWidgets.QComboBox()
                for key, label in EMIT_LABELS.items():
                    combo.addItem(label, key)
                combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
                self.table.setCellWidget(r, C_EMIT, combo)
            combo.blockSignals(True)
            combo.setCurrentIndex(list(EMIT_LABELS).index(b.emit))
            combo.blockSignals(False)
            self._set(r, C_NAME, b.name)
            for c, v in ((C_X1, b.x1), (C_Y1, b.y1), (C_X2, b.x2), (C_Y2, b.y2), (C_ANG, b.angle_deg),
                         (C_LEN, b.length), (C_THK, b.thickness), (C_AMP, b.amplitude),
                         (C_PH, math.degrees(b.phase))):
                self._set(r, c, _fmt(v))
        finally:
            self._updating = False

    def _num(self, r: int, c: int) -> float:
        item = self.table.item(r, c)
        text = "" if item is None else item.text().strip()
        try:
            return float(text)
        except ValueError:
            raise ConfigError(f"막대 {r + 1} '{COLS[c]}': 숫자가 아닙니다 ({text!r}).") from None

    def _read_row(self, r: int) -> Bar:
        if self.table.cellWidget(r, C_EMIT) is None:
            raise ConfigError(f"막대 {r + 1}: 아직 입력 중입니다.")
        name_item = self.table.item(r, C_NAME)
        return Bar(self._num(r, C_X1), self._num(r, C_Y1), self._num(r, C_X2), self._num(r, C_Y2),
                   self._num(r, C_THK), self.table.cellWidget(r, C_EMIT).currentData(), self._num(r, C_AMP),
                   math.radians(self._num(r, C_PH)), name_item.text() if name_item else "")

    def get_bars(self) -> tuple[Bar, ...]:
        if self.table.rowCount() == 0:
            raise ConfigError("막대가 없습니다. [추가] 또는 [예시 불러오기] 를 누르세요.")
        return tuple(self._read_row(r) for r in range(self.table.rowCount()))

    def try_bars(self) -> tuple[Bar, ...]:
        """미리보기용: 읽을 수 있는 행만 (입력 중인 잘못된 칸은 건너뜀)."""
        out = []
        for r in range(self.table.rowCount()):
            try:
                b = self._read_row(r)
                if b.length > 0 and b.thickness > 0:
                    out.append(b)
            except ConfigError:
                continue
        return tuple(out)

    def set_bars(self, bars) -> None:
        """표 전체를 바꾼다. 채우는 동안 신호를 막고 끝에 한 번만 알린다 (반쯤 채운 행을 읽지 않게)."""
        self.table.blockSignals(True)
        try:
            self.table.clearSelection()
            self.table.setRowCount(0)
            for b in bars:
                self.table.insertRow(self.table.rowCount())
                self._write_row(self.table.rowCount() - 1, b)
        finally:
            self.table.blockSignals(False)
        self.changed.emit()
        self.selectionChanged.emit(self.current_row())

    # ------------------------------------------------------------------ 끝점 ↔ 각도·길이 동기화
    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating:
            return
        r, c = item.row(), item.column()
        try:
            if c in (C_X1, C_Y1, C_X2, C_Y2):  # 끝점 → 각도·길이
                x1, y1, x2, y2 = (self._num(r, k) for k in (C_X1, C_Y1, C_X2, C_Y2))
                self._updating = True
                self._set(r, C_ANG, _fmt(math.degrees(math.atan2(y2 - y1, x2 - x1))))
                self._set(r, C_LEN, _fmt(math.hypot(x2 - x1, y2 - y1)))
            elif c in (C_ANG, C_LEN):  # 끝점 1 + 각도·길이 → 끝점 2
                x1, y1, ang, ln = (self._num(r, k) for k in (C_X1, C_Y1, C_ANG, C_LEN))
                self._updating = True
                self._set(r, C_X2, _fmt(x1 + ln * math.cos(math.radians(ang))))
                self._set(r, C_Y2, _fmt(y1 + ln * math.sin(math.radians(ang))))
        except ConfigError:
            pass  # 입력 중 (숫자가 아직 아님) — 계산 실행 때 오류로 알린다
        finally:
            self._updating = False
        self.changed.emit()

    # ------------------------------------------------------------------ 버튼
    def add_bar(self) -> None:
        if self.table.rowCount() >= MAX_BARS:
            QtWidgets.QMessageBox.information(self, "막대 개수", f"막대는 최대 {MAX_BARS}개입니다.")
            return
        r = self.current_row() + 1 if self.current_row() >= 0 else self.table.rowCount()
        self.table.insertRow(r)
        self._write_row(r, Bar(0.5, 0.0, 0.5, 0.5, 0.06, "none", 1e-5, 0.0, f"막대 {r + 1}"))
        self.table.selectRow(r)
        self.changed.emit()

    def duplicate(self) -> None:
        r = self.current_row()
        if r < 0 or self.table.rowCount() >= MAX_BARS:
            return
        try:
            b = self._read_row(r)
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "복제 불가", str(e))
            return
        shift = 0.1
        self.table.insertRow(r + 1)
        self._write_row(r + 1, Bar(b.x1, b.y1 - shift, b.x2, b.y2 - shift, b.thickness, b.emit, b.amplitude,
                                   b.phase, (b.name or f"막대 {r + 1}") + " 복사"))
        self.table.selectRow(r + 1)
        self.changed.emit()

    def remove(self) -> None:
        r = self.current_row()
        if r >= 0:
            self.table.removeRow(r)
            self.changed.emit()

    def move(self, step: int) -> None:
        r = self.current_row()
        t = r + step
        if r < 0 or not (0 <= t < self.table.rowCount()):
            return
        try:
            a, b = self._read_row(r), self._read_row(t)
        except ConfigError:
            return
        self._write_row(r, b)
        self._write_row(t, a)
        self.table.selectRow(t)
        self.changed.emit()

    def flip(self) -> None:
        """끝점을 뒤집으면 왼쪽/오른쪽이 바뀌므로 방사면도 바꿔 같은 물리적 면을 유지한다."""
        r = self.current_row()
        if r < 0:
            return
        try:
            b = self._read_row(r)
        except ConfigError:
            return
        swap = {"left": "right", "right": "left"}
        self._write_row(r, Bar(b.x2, b.y2, b.x1, b.y1, b.thickness, swap.get(b.emit, b.emit), b.amplitude,
                               b.phase, b.name))
        self.changed.emit()

    def load_preset(self, key: str) -> None:
        pr: FramePreset = self._presets[key]
        self.preset_box.setCurrentIndex(self.preset_box.findData(key))
        self.set_bars(pr.bars)
        self.presetLoaded.emit(pr)

    # ------------------------------------------------------------------ 파일
    def to_json(self) -> list[dict]:
        return [dict(name=b.name, x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, thickness=b.thickness, emit=b.emit,
                     amplitude=b.amplitude, phase_deg=math.degrees(b.phase)) for b in self.get_bars()]

    @staticmethod
    def bars_from_json(data) -> tuple[Bar, ...]:
        items = data["bars"] if isinstance(data, dict) else data
        out = []
        for i, d in enumerate(items):
            try:
                out.append(Bar(float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]),
                               float(d["thickness"]), d.get("emit", "none"), float(d.get("amplitude", 1e-5)),
                               math.radians(float(d.get("phase_deg", 0.0))), d.get("name", "")))
            except (KeyError, TypeError, ValueError) as e:
                raise ConfigError(f"파일의 막대 {i + 1}: 형식 오류 ({e}).") from None
            if out[-1].emit not in EMIT_SIDES:
                raise ConfigError(f"파일의 막대 {i + 1}: 방사면 값 {out[-1].emit!r} 이 잘못되었습니다.")
        return tuple(out)

    def save_to(self, path: Path, extra: dict | None = None) -> None:
        data = {"bars": self.to_json(), **(extra or {})}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_dialog(self) -> None:
        try:
            self.to_json()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "저장 불가", str(e))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "프레임 저장", "frame.json", "JSON (*.json)")
        if path:
            try:
                self.save_to(Path(path), self.window_extra())
            except PermissionError:
                QtWidgets.QMessageBox.warning(self, "저장 실패", f"쓰기 권한이 없거나 파일이 열려 있습니다:\n{path}")

    def load_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "프레임 불러오기", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.set_bars(self.bars_from_json(data))
            if isinstance(data, dict):
                self.presetLoaded.emit(data)  # 영역·기준점이 저장돼 있으면 함께 적용
        except (OSError, json.JSONDecodeError, ConfigError) as e:
            QtWidgets.QMessageBox.warning(self, "불러오기 실패", str(e))

    def window_extra(self) -> dict:
        """저장 파일에 함께 넣을 값 (탭이 덮어씀: 영역, 기준점 등)."""
        return {}
