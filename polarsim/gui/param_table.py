"""행 = 객체 하나인 편집 표 (물체·송신기 공용). 열 종류: 숫자 / 글자 / 선택 / 체크.

값 읽기는 row_values() → {key: 값} (숫자는 float). 잘못된 숫자는 '몇 번째 행 어느 열' 을 알려 주는 ConfigError.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from ..config import ConfigError


@dataclass(frozen=True)
class Column:
    key: str
    header: str
    kind: str = "float"  # float | text | combo | check
    default: object = 0.0
    options: tuple = ()  # combo: ((값, 표시), ...)
    tip: str = ""
    width: int = 58


class ParamTable(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, columns: list[Column], noun: str, max_rows: int, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.noun = noun
        self.max_rows = max_rows
        self.table = QtWidgets.QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels([c.header for c in columns])
        for k, c in enumerate(columns):
            item = self.table.horizontalHeaderItem(k)
            if c.tip:
                item.setToolTip(c.tip)
            self.table.setColumnWidth(k, c.width)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(150)  # 행이 적을 때 빈 공간이 커지지 않게 (아래 [계산 실행] 이 화면에 보이도록)
        self.table.itemChanged.connect(lambda _it: self._emit())
        self.buttons = QtWidgets.QHBoxLayout()
        for text, fn in (("추가", self.add_row), ("복제", self.duplicate), ("삭제", self.remove)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            self.buttons.addWidget(b)
        self.buttons.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.table)
        lay.addLayout(self.buttons)
        self._block = False

    def _emit(self) -> None:
        if not self._block:
            self.changed.emit()

    # ------------------------------------------------------------------ 쓰기
    def _write_row(self, r: int, values: dict) -> None:
        for k, c in enumerate(self.columns):
            v = values.get(c.key, c.default)
            if c.kind == "combo":
                box = QtWidgets.QComboBox()
                for val, label in c.options:
                    box.addItem(label, val)
                i = max(0, box.findData(v))
                box.setCurrentIndex(i)
                box.currentIndexChanged.connect(lambda _i: self._emit())
                self.table.setCellWidget(r, k, box)
            elif c.kind == "check":
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
                              | QtCore.Qt.ItemFlag.ItemIsSelectable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if v else QtCore.Qt.CheckState.Unchecked)
                self.table.setItem(r, k, item)
            else:
                text = f"{v:.6g}" if isinstance(v, float) else str(v)
                self.table.setItem(r, k, QtWidgets.QTableWidgetItem(text))

    def set_rows(self, rows: list[dict]) -> None:
        self._block = True
        try:
            self.table.setRowCount(0)
            for r, values in enumerate(rows[: self.max_rows]):
                self.table.insertRow(r)
                self._write_row(r, values)
        finally:
            self._block = False
        self.changed.emit()

    def add_row(self, values: dict | None = None) -> None:
        if self.table.rowCount() >= self.max_rows:
            QtWidgets.QMessageBox.information(self, self.noun, f"{self.noun}는 최대 {self.max_rows}개입니다.")
            return
        values = values or {}
        r = self.table.rowCount()
        self._block = True
        try:
            self.table.insertRow(r)
            self._write_row(r, values)
        finally:
            self._block = False
        self.table.selectRow(r)
        self.changed.emit()

    def current_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else self.table.rowCount() - 1

    def duplicate(self) -> None:
        r = self.current_row()
        if r < 0:
            return
        values = self.raw_row(r)
        if "name" in values:
            values["name"] = f"{values['name']} (복사)"
        self.add_row(values)

    def remove(self) -> None:
        r = self.current_row()
        if r < 0:
            return
        self.table.removeRow(r)
        self.changed.emit()

    # ------------------------------------------------------------------ 읽기
    def raw_row(self, r: int) -> dict:
        """표에 적힌 그대로 (숫자 칸도 글자)."""
        out = {}
        for k, c in enumerate(self.columns):
            if c.kind == "combo":
                w = self.table.cellWidget(r, k)
                out[c.key] = w.currentData() if w is not None else c.default
            elif c.kind == "check":
                it = self.table.item(r, k)
                out[c.key] = it is not None and it.checkState() == QtCore.Qt.CheckState.Checked
            else:
                it = self.table.item(r, k)
                out[c.key] = it.text().strip() if it is not None else ""
        return out

    def row_values(self, r: int) -> dict:
        raw = self.raw_row(r)
        for k, c in enumerate(self.columns):
            if c.kind == "float":
                try:
                    raw[c.key] = float(raw[c.key])
                except ValueError:
                    raise ConfigError(f"{self.noun} {r + 1}행 '{c.header}': 숫자가 아닙니다 ({raw[c.key]!r}).") from None
        return raw

    def all_values(self) -> list[dict]:
        return [self.row_values(r) for r in range(self.table.rowCount())]
