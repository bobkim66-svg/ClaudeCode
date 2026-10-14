"""실행 진입점.

- 로그: Windows 는 %LOCALAPPDATA%\\polarsim\\logs\\polarsim.log, 그 외 ~/.local/state/polarsim/logs.
  --noconsole 빌드에서는 콘솔이 없으므로 모든 예외를 로그에 남기고 대화상자로 로그 위치를 알린다.
- 리소스: PyInstaller 번들(sys._MEIPASS) 과 소스 실행 모두에서 docs/ 를 찾는다.
- --selftest OUT_DIR: 창을 띄워 프리셋 3종 + 엔진 B + 3D 소나를 계산·저장하고 물리 검증 결과를 OUT_DIR 에 기록,
  실패가 있으면 종료 코드 1. CI(Windows 러너) 에서 exe 단독 실행 확인에 사용.
- --software-gl: QT_OPENGL=software (원격 데스크톱처럼 OpenGL 2.0 이 없는 환경에서 3D 화면용).
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
SOFTWARE_GL_STATUS = "해당 없음"  # main() 에서 설정 (자체진단 보고용)
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
    """물리 검증 + GUI 전체 경로 (프리셋 계산, 애니메이션, 저장) 를 실행하고 보고서를 남긴다.

    어떤 예외도 밖으로 내보내지 않는다: --noconsole exe 에서 최상위 예외는 PyInstaller 가 모달 대화상자로 띄워
    CI 에서 영원히 멈추기 때문. 예외는 로그와 보고서에 남기고 종료 코드 1."""
    try:
        return _run_selftest_body(out_dir, log_path)
    except BaseException:  # 진단 결과로 기록 (삼키는 것이 아니라 보고서에 FAIL 로 남김)
        text = traceback.format_exc()
        log.critical("selftest aborted:\n%s", text)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "selftest_report.txt").write_text(f"FAIL  자체진단 중단 (예외)\n{text}", encoding="utf-8")
        return 1


def _run_selftest_body(out_dir: Path, log_path: Path) -> int:
    import numpy as np

    from .config import SimConfig, Source
    from .gui.main_window import MainWindow
    from .metrics import simulate_points
    from .presets import PRESETS

    import faulthandler

    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        log.info("selftest %s: %s %s", "PASS" if ok else "FAIL", name, detail)

    # 자체진단은 사람이 없는 CI 에서도 돈다: 처리 안 된 예외로 모달 대화상자가 뜨면 영원히 멈추므로 기록만 한다.
    unhandled: list[str] = []

    def record(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("selftest unhandled exception:\n%s", text)
        unhandled.append(f"{exc_type.__name__}: {exc}")

    sys.excepthook = record
    threading.excepthook = lambda a: record(a.exc_type, a.exc_value, a.exc_traceback)
    hang_file = (out_dir / "hang_traceback.txt").open("w", encoding="utf-8")
    faulthandler.dump_traceback_later(600, exit=True, file=hang_file)  # 10 분 넘게 멈추면 스택을 남기고 종료

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
    log.info("selftest: 창 만들기")
    win = MainWindow(resource_path(GUIDE_REL), log_path)
    win.resize(1600, 1000)
    log.info("selftest: 창 표시")
    win.show()
    app.processEvents()
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
    win.tabs.setCurrentWidget(win.wave_tab)
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
    _selftest_3d(win, app, out_dir, check)
    win.tabs.setCurrentWidget(win.help)
    app.processEvents()
    win.grab().save(str(out_dir / "screenshot_help.png"))
    win.close()

    faulthandler.cancel_dump_traceback_later()
    hang_file.close()
    check("처리되지 않은 예외 없음", not unhandled, "; ".join(unhandled[:3]))
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}  {detail}" for name, ok, detail in checks]
    n_fail = sum(not ok for _, ok, _ in checks)
    lines.append(f"결과: {len(checks) - n_fail}/{len(checks)} 통과, 로그 {log_path}")
    (out_dir / "selftest_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("selftest:\n%s", "\n".join(lines))
    return 1 if n_fail else 0


def _selftest_3d(win, app, out_dir: Path, check) -> None:
    """3D 소나 탭: 작은 격자로 AUV 예시 계산, 검증값, 저장, 그리고 3D 화면이 실제로 그려지는지 (OpenGL)."""
    import numpy as np

    t3 = win.tab3d
    log.info("selftest 3D: 탭 전환")
    win.tabs.setCurrentWidget(t3)
    app.processEvents()
    log.info("selftest 3D: 예시 불러오기")
    t3.load_preset("auv_cw")
    t3.cpw.setValue(8)  # 자체진단 시간 단축
    log.info("selftest 3D: 계산")
    t3.run_sync()
    t3.timer.stop()
    log.info("selftest 3D: 재생")
    for _ in range(5):
        t3._tick()
        app.processEvents()
    r = t3.res
    check("[3D] 회전 검증 (float32 반올림 수준)", r.curl_ratio < 1e-4, f"curl={r.curl_ratio:.1e}")
    check("[3D] 연속파 정상상태 도달", r.convergence < 0.02, f"창 차이={r.convergence:.1e}, {r.solve_s:.1f} s")
    log.info("selftest 3D: 화면 캡처")
    img = t3.view.grab_image()
    renderer = "?"
    try:
        from OpenGL import GL

        t3.view.view.makeCurrent()
        renderer = (GL.glGetString(GL.GL_RENDERER) or b"?").decode(errors="replace") + " / " + \
            (GL.glGetString(GL.GL_VERSION) or b"?").decode(errors="replace")
    except Exception as e:  # 진단 정보일 뿐 — 실패 판정은 아래 픽셀 검사로 한다
        renderer = f"조회 실패: {type(e).__name__}"
    if t3.view.view.gl_error:
        renderer = f"OpenGL 실패: {t3.view.view.gl_error}"
    drawn = 0.0
    if not img.isNull():
        from .gui.view3d import BG

        ptr = img.convertToFormat(img.Format.Format_RGBA8888)
        arr = np.frombuffer(ptr.constBits(), np.uint8).reshape(ptr.height(), ptr.width(), 4)[..., :3]
        drawn = float(np.mean(np.any(np.abs(arr.astype(int) - np.array(BG)) > 12, axis=-1)))
    bundle = Path(getattr(sys, "_MEIPASS", ""))
    sw = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("opengl32sw.dll")) if bundle.name else []
    check("[3D] 소프트웨어 GL 연결 상태 (정보)", True, SOFTWARE_GL_STATUS)
    check("[3D] 3D 화면 렌더링 (OpenGL)", drawn > 0.05,
          f"그려진 픽셀 {drawn:.0%}, {renderer}, QT_OPENGL={os.environ.get('QT_OPENGL', '(기본)')}, "
          f"소프트웨어 GL DLL: {sw or '없음/소스 실행'}")
    saved = t3.export_to(out_dir)
    names = sorted(p.name for p in saved.iterdir())
    check("[3D] 결과 저장", {"config.json", "summary.html", "particles.csv", "view3d.png"} <= set(names),
          ", ".join(names))
    win.grab().save(str(out_dir / "screenshot_3d.png"))


def pyopengl_follow_qt_software_gl() -> str:
    """QT_OPENGL=software 일 때 PyOpenGL 도 Qt 와 같은 소프트웨어 GL (opengl32sw.dll) 을 쓰게 한다 (Windows).

    PyOpenGL 은 이름으로 시스템 opengl32.dll 을 불러오는데, Qt 의 소프트웨어 모드는 opengl32sw.dll (Mesa) 에서
    컨텍스트를 만든다. 둘이 다르면 pyqtgraph 의 GL 호출이 '현재 컨텍스트 없음' (GLError 1282) 이 된다 (CI 실측).
    OpenGL.GL 을 import 하기 전에 불러야 한다 (GL 함수 객체가 그때 라이브러리에 묶임). 반환: 상태 설명 (로그용)."""
    if sys.platform != "win32" or os.environ.get("QT_OPENGL") != "software":
        return "해당 없음"
    if "OpenGL.GL" in sys.modules:
        return "늦음: OpenGL.GL 이 이미 import 됨"
    import ctypes

    import PySide6

    candidates = [Path(PySide6.__file__).resolve().parent / "opengl32sw.dll"]
    if hasattr(sys, "_MEIPASS"):
        candidates += sorted(Path(sys._MEIPASS).rglob("opengl32sw.dll"))
    dll = next((c for c in candidates if c.is_file()), None)
    if dll is None:
        return "opengl32sw.dll 없음"
    import OpenGL.platform as plat  # 여기서 시스템 opengl32 를 한 번 불러오지만 아래에서 바꿔 끼운다

    from OpenGL.platform import baseplatform

    lib = ctypes.WinDLL(str(dll), mode=ctypes.RTLD_GLOBAL)
    P = plat.PLATFORM
    # import 때 install() 이 시스템 opengl32 로 계산해 캐시한 값 (GetCurrentContext, CurrentContextIsValid,
    # getExtensionProcedure, WGL 등) 을 모두 지우고, 새 라이브러리로 다시 install 한다.
    for name in list(vars(P)):
        if isinstance(getattr(type(P), name, None), baseplatform.lazy_property):
            delattr(P, name)
    P.GL = lib  # lazy_property 는 인스턴스 속성이 있으면 그것을 쓴다
    P.OpenGL = lib
    P.install(vars(plat))
    addr = lambda f: ctypes.cast(f, ctypes.c_void_p).value  # noqa: E731
    same = addr(plat.GetCurrentContext) == addr(lib.wglGetCurrentContext)
    return f"PyOpenGL → {dll}; GetCurrentContext 가 opengl32sw 함수: {same}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("--selftest", type=Path, metavar="OUT_DIR", help="자체 진단 후 종료")
    parser.add_argument("--software-gl", action="store_true",
                        help="3D 화면을 소프트웨어 OpenGL 로 그림 (원격 데스크톱 등에서 3D 화면이 비거나 검을 때)")
    args = parser.parse_args(argv)
    flag = log_dir().parent / "use_software_gl"  # 3D 화면 OpenGL 실패 시 자동으로 생기는 표시 (gui/view3d.py)
    if args.software_gl or flag.exists():
        os.environ["QT_OPENGL"] = "software"  # QApplication 만들기 전에 정해야 함
    log_path = setup_logging()
    install_excepthook(log_path)
    log.info("start %s, python %s, frozen=%s, QT_OPENGL=%s", APP_NAME, sys.version.split()[0],
             getattr(sys, "frozen", False), os.environ.get("QT_OPENGL", "(기본)"))
    global SOFTWARE_GL_STATUS
    try:
        SOFTWARE_GL_STATUS = pyopengl_follow_qt_software_gl()
    except Exception as e:  # 실패해도 앱은 뜬다 (3D 화면은 안내문으로 대체됨)
        SOFTWARE_GL_STATUS = f"전환 실패: {type(e).__name__}: {e}"
        log.exception("PyOpenGL 소프트웨어 GL 전환 실패")
    log.info("software GL: %s", SOFTWARE_GL_STATUS)
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
