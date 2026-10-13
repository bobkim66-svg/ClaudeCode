"""소스 편집 표. 사용자는 위상을 도(°) 로 입력하고, 여기서 한 번만 rad 로 변환한다."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from ..config import MAX_SOURCES, ConfigError, Source

COLUMNS = ["켜기", "종류", "방향 x", "방향 y", "방향 z", "진폭", "주파수", "위상 [°]",
           "허리 w0", "기준점 x", "기준점 y", "기준점 z"]
KIND_LABELS = {"plane": "평면파", "gaussian": "가우시안 빔"}
C_ON, C_KIND, C_DX, C_DY, C_DZ, C_AMP, C_FREQ, C_PHASE, C_WAIST, C_RX, C_RY, C_RZ = range(len(COLUMNS))


class SourceTable(QtWidgets.QWidget):
    """소스 1~MAX_SOURCES 개 편집. get_sources() 가 검증 가능한 Source 튜플을 만든다."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.itemChanged.connect(lambda _item: self.changed.emit())
        add = QtWidgets.QPushButton("소스 추가")
        rem = QtWidgets.QPushButton("선택 소스 삭제")
        add.clicked.connect(self._add_default)
        rem.clicked.connect(self._remove_selected)
        self.unit_label = QtWidgets.QLabel()
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(add)
        btns.addWidget(rem)
        btns.addStretch(1)
        btns.addWidget(self.unit_label)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.table)
        lay.addLayout(btns)
        self.set_units(dimensionless=False)

    # ---- 단위 표시 ----------------------------------------------------------
    def set_units(self, dimensionless: bool) -> None:
        if dimensionless:
            self.unit_label.setText("단위: 무차원 (c=1, 파장=1/f), 위상 [°]")
        else:
            self.unit_label.setText("단위: 진폭·w0·기준점 [m], 주파수 [Hz], 위상 [°]")

    # ---- 행 편집 ------------------------------------------------------------
    def _add_default(self) -> None:
        if self.table.rowCount() >= MAX_SOURCES:
            QtWidgets.QMessageBox.information(self, "소스 개수", f"소스는 최대 {MAX_SOURCES}개입니다.")
            return
        prev = self.table.rowCount()
        freq = float(self.table.item(prev - 1, C_FREQ).text()) if prev else 1000.0
        self._append(Source(direction=(0.0, 1.0, 0.0), frequency=freq))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            if self.table.rowCount() > 1:
                self.table.removeRow(r)
        self.changed.emit()

    def _append(self, s: Source) -> None:
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        on = QtWidgets.QTableWidgetItem()
        on.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        on.setCheckState(QtCore.Qt.CheckState.Checked if s.enabled else QtCore.Qt.CheckState.Unchecked)
        self.table.setItem(r, C_ON, on)
        kind = QtWidgets.QComboBox()
        for key, label in KIND_LABELS.items():
            kind.addItem(label, key)
        kind.setCurrentIndex(list(KIND_LABELS).index(s.kind))
        kind.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.table.setCellWidget(r, C_KIND, kind)
        waist = "" if s.waist is None else f"{s.waist:.6g}"
        values = {
            C_DX: s.direction[0], C_DY: s.direction[1], C_DZ: s.direction[2],
            C_AMP: s.amplitude, C_FREQ: s.frequency, C_PHASE: math.degrees(s.phase),
            C_RX: s.ref_point[0], C_RY: s.ref_point[1], C_RZ: s.ref_point[2],
        }
        for col, v in values.items():
            self.table.setItem(r, col, QtWidgets.QTableWidgetItem(f"{float(v):.6g}"))
        self.table.setItem(r, C_WAIST, QtWidgets.QTableWidgetItem(waist))
        self.table.blockSignals(False)

    def set_sources(self, sources) -> None:
        self.table.setRowCount(0)
        for s in sources:
            self._append(s)
        self.changed.emit()

    # ---- 읽기 ---------------------------------------------------------------
    def _num(self, row: int, col: int, allow_empty: bool = False) -> float | None:
        item = self.table.item(row, col)
        text = "" if item is None else item.text().strip()
        if text == "" and allow_empty:
            return None
        try:
            return float(text)
        except ValueError:
            raise ConfigError(f"소스 {row + 1} '{COLUMNS[col]}': 숫자가 아닙니다 ({text!r}).") from None

    def get_sources(self) -> tuple[Source, ...]:
        out = []
        for r in range(self.table.rowCount()):
            kind = self.table.cellWidget(r, C_KIND).currentData()
            out.append(Source(
                direction=(self._num(r, C_DX), self._num(r, C_DY), self._num(r, C_DZ)),
                amplitude=self._num(r, C_AMP),
                frequency=self._num(r, C_FREQ),
                phase=math.radians(self._num(r, C_PHASE)),
                enabled=self.table.item(r, C_ON).checkState() == QtCore.Qt.CheckState.Checked,
                kind=kind,
                waist=self._num(r, C_WAIST, allow_empty=True),
                ref_point=(self._num(r, C_RX), self._num(r, C_RY), self._num(r, C_RZ)),
            ))
        return tuple(out)
