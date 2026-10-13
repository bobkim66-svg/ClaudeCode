# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 빌드 설정. 사용: pyinstaller polarsim.spec  (onefile 은 환경변수 POLARSIM_ONEFILE=1)
import os

from PyInstaller.utils.hooks import collect_submodules

ONEFILE = os.environ.get("POLARSIM_ONEFILE") == "1"

# pyqtgraph.opengl 은 '3D 궤적 창' 버튼에서 지연 import 되므로 명시적으로 포함한다.
hidden = collect_submodules("pyqtgraph.opengl") + ["OpenGL.platform.win32", "OpenGL.arrays.numpymodule"]

# 쓰지 않는 무거운 모듈 제외 (용량/오탐 감소). PySide6 는 Essentials 만 설치하므로 WebEngine 등은 애초에 없음.
excludes = ["tkinter", "matplotlib", "IPython", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtQuick",
            "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.Qt3DCore"]

a = Analysis(
    ["run_polarsim.py"],
    pathex=[],
    binaries=[],
    datas=[("docs/USER_GUIDE_ko.md", "docs")],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="polarsim", console=False, upx=False,
              runtime_tmpdir=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="polarsim", console=False, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="polarsim", upx=False)
