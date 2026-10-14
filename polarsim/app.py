"""실행 진입점.

- 로그: Windows 는 %LOCALAPPDATA%\\polarsim\\logs\\polarsim.log, 그 외 ~/.local/state/polarsim/logs.
  --noconsole 빌드에서는 콘솔이 없으므로 모든 예외를 로그에 남기고 대화상자로 로그 위치를 알린다.
- 리소스: PyInstaller 번들(sys._MEIPASS) 과 소스 실행 모두에서 docs/ 를 찾는다.
- --selftest OUT_DIR: 창을 띄워 프리셋 3종을 계산·저장하고 물리 검증 결과를 OUT_DIR 에 기록,
  실패가 있으면 종료 코드 1. CI(Windows 러너) 에서 exe 단독 실행 확인에 사용.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path

APP_NAME = "polarsim"
GUIDE_REL = Path("docs") / "USER_GUIDE_ko.md"
log = logging.getLogger(APP_NAME)


def resource_path(rel: Path) -> Path:
    """번들 내부(_MEIPASS) 또는 저장소 루트 기준 경로."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / rel


def log_dir() -> Path:
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME / "logs"
    return Path.home() / ".local" / "state" / APP_NAME / "logs"


def setup_logging() -> Path:
    d = log_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "polarsim.log"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    ]
    if sys.stderr is not None:  # --noconsole 빌드에서는 None
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    logging.captureWarnings(True)
    return path


def install_excepthook(log_path: Path) -> None:
    """처리되지 않은 예외 → 로그 + (GUI 가 있으면) 로그 위치를 알리는 대화상자. 예외를 삼키지 않는다."""

    def show(exc_type, exc, tb):
        log.critical("unhandled exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        from PySide6 import QtWidgets

        if QtWidgets.QApplication.instance() is not None and threading.current_thread() is threading.main_thread():
            QtWidgets.QMessageBox.critical(
                None, "오류", f"예기치 못한 오류가 발생했습니다:\n{exc_type.__name__}: {exc}\n\n로그 파일:\n{log_path}")

    sys.excepthook = show
    threading.excepthook = lambda a: show(a.exc_type, a.exc_value, a.exc_traceback)


def pick_font_family(families: list[str]) -> str | None:
    """한글 표시 가능한 폰트 우선순위."""
    for name in ("Malgun Gothic", "맑은 고딕", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Noto Serif CJK KR"):
        if name in families:
            return name
    return None


def create_app(argv: list[str]):
    from PySide6 import QtCore, QtGui, QtWidgets

    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    family = pick_font_family(QtGui.QFontDatabase.families())
    if family:
        font = QtGui.QFont(family)
        font.setPointSize(9)
        app.setFont(font)
    else:
        log.warning("한글 폰트를 찾지 못했습니다. 글자가 깨질 수 있습니다.")
    log.info("font=%s, platform=%s, dpr=%.2f", family, app.platformName(), app.devicePixelRatio())
    from .gui.style import apply_global_style

    apply_global_style()
    return app


def run_selftest(out_dir: Path, log_path: Path) -> int:
    """물리 검증 + GUI 전체 경로 (프리셋 계산, 애니메이션, 저장) 를 실행하고 보고서를 남긴다."""
    import numpy as np

    from .config import SimConfig, Source
    from .gui.main_window import MainWindow
    from .metrics import simulate_points
    from .presets import PRESETS

    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    lam = 343.0 / 1000.0
    r = simulate_points(SimConfig(sources=(Source(direction=(1, 2, 3)),)), np.zeros((1, 3)))
    check("단일 평면파 L=1, T=0", abs(r.L[0] - 1) < 1e-9 and abs(r.T[0]) < 1e-9, f"L={r.L[0]}, T={r.T[0]}")
    r = simulate_points(PRESETS["circular"].config, np.zeros((1, 3)))
    check("직교 2파 원점 e=1, T=0.5", abs(r.e[0] - 1) < 1e-9 and abs(r.T[0] - 0.5) < 1e-9, f"e={r.e[0]}, T={r.T[0]}")
    y0 = 2 * lam / (3 * np.sqrt(3))
    r = simulate_points(PRESETS["hex"].config, np.array([[0, y0, 0]]))
    check("120° 3파 소용돌이 중심 e=1, T=NaN", abs(r.e[0] - 1) < 1e-9 and np.isnan(r.T[0]), f"e={r.e[0]}")

    try:  # '3D 궤적 창' 은 지연 import 이므로 번들 누락을 여기서 미리 확인 (GL 컨텍스트 생성은 하지 않음)
        import pyqtgraph.opengl  # noqa: F401

        check("OpenGL 3D 모듈 포함", True)
    except Exception as e:
        check("OpenGL 3D 모듈 포함", False, f"{type(e).__name__}: {e}")

    app = create_app([sys.argv[0]])
    win = MainWindow(resource_path(GUIDE_REL), log_path)
    win.resize(1600, 1000)
    win.show()
    check("도움말 가이드 로드", "찾을 수 없습니다" not in win.help.toPlainText(), str(resource_path(GUIDE_REL)))
    for key in ("standing", "circular", "hex"):
        win.load_preset(key)
        win.run_sync(win.build_config())
        for _ in range(5):
            win._tick()
            app.processEvents()
        res = win.bundle.result
        check(f"[{key}] GUI 계산·회전 검증", res.curl_ratio < 1e-4, f"curl={res.curl_ratio:.2e}")
        if key == "standing":
            check("[standing] T 전 영역 NaN", bool(np.all(np.isnan(res.T))))
        win.flow.setChecked(key == "hex")
        app.processEvents()
        saved = win.export_to(out_dir)
        files = sorted(p.name for p in saved.iterdir())
        check(f"[{key}] 결과 저장", len(files) == 6, ", ".join(files))
        win.grab().save(str(out_dir / f"screenshot_{key}.png"))
    # 엔진 B: AUV 품 방사 (기본 형상)
    win.tabs.setCurrentIndex(1)
    wt = win.wave_tab
    wt.run_sync()
    for _ in range(5):
        wt._tick()
        app.processEvents()
    b = wt.bundle
    check("[엔진 B] 회전 검증 (구조적 0)", b.res.curl_ratio < 1e-10, f"curl={b.res.curl_ratio:.1e}")
    ratio = b.power_out / b.power_in if b.power_out else float("nan")
    check("[엔진 B] 에너지 보존 (방사 일률 = 유출 세기)", abs(ratio - 1) < 0.03, f"ratio={ratio:.4f}")
    e_far = [r["e_energy"] for r in b.radial if r["n"] > 0 and r["coverage"] > 0.95]  # 영역이 다 덮는 구간만
    check("[엔진 B] 품 밖 타원도가 거리에 따라 감소", len(e_far) >= 3 and e_far[-1] < e_far[1],
          ", ".join(f"{v:.3f}" for v in e_far))
    saved = wt.export_to(out_dir)
    check("[엔진 B] 결과 저장", len(list(saved.iterdir())) == 7, ", ".join(sorted(p.name for p in saved.iterdir())))
    win.grab().save(str(out_dir / "screenshot_engineB.png"))
    win.tabs.setCurrentIndex(2)
    app.processEvents()
    win.grab().save(str(out_dir / "screenshot_help.png"))
    win.close()

    lines = [f"{'PASS' if ok else 'FAIL'}  {name}  {detail}" for name, ok, detail in checks]
    n_fail = sum(not ok for _, ok, _ in checks)
    lines.append(f"결과: {len(checks) - n_fail}/{len(checks)} 통과, 로그 {log_path}")
    (out_dir / "selftest_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("selftest:\n%s", "\n".join(lines))
    return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("--selftest", type=Path, metavar="OUT_DIR", help="자체 진단 후 종료")
    args = parser.parse_args(argv)
    log_path = setup_logging()
    install_excepthook(log_path)
    log.info("start %s, python %s, frozen=%s", APP_NAME, sys.version.split()[0], getattr(sys, "frozen", False))
    if args.selftest:
        return run_selftest(args.selftest, log_path)

    from .gui.main_window import MainWindow

    app = create_app(sys.argv)
    win = MainWindow(resource_path(GUIDE_REL), log_path)
    win.resize(1600, 1000)
    win.show()
    win.load_preset("circular", run=True)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
