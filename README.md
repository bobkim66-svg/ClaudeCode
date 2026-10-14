# polarsim — 다중 음파 중첩에 의한 입자 운동 편광 시뮬레이터

- **3D 소나 탭**: 수중 소나 송신기·AUV·해저 물체가 있는 3D 공간에서 물 입자 움직임과 음압을 3D FDTD 로 계산하고,
  CAD 처럼 돌려 보는 3D 뷰(단면, 입자 애니메이션, STL 불러오기)로 본다. 연속파와 핑(펄스) 모두 지원.
- 엔진 A/B: 서로 다른 방향의 음파(종파)를 중첩했을 때 입자 궤적이 직선에서 타원/원으로 바뀌는 정도를 정량화한다 (2D·해석해).

유체에는 횡파가 생기지 않으며 변위장은 항상 비회전(∇×u = 0)이다. 자세한 내용은 [사용자 가이드](docs/USER_GUIDE_ko.md).

## 실행 파일 받기 (Windows)

GitHub 저장소의 **Actions → build-windows → 최근 성공한 실행 → Artifacts** 에서
`polarsim-windows-onedir` 를 내려받아 압축을 풀고 `polarsim.exe` 를 실행한다.
(`polarsim-selftest` 에는 Windows 러너에서 exe 를 단독 실행한 자체진단 보고서와 화면 캡처가 들어 있다.)

## 소스에서 실행 / 테스트 (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe run_polarsim.py
.\.venv\Scripts\python.exe -m pytest -q
```

## 직접 빌드 (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1            # dist\polarsim\polarsim.exe (폴더 형식, 기본)
powershell -ExecutionPolicy Bypass -File .\build.ps1 -OneFile   # dist\polarsim.exe (단일 파일, 옵션)
```

## 구조

| 경로 | 역할 |
|---|---|
| `polarsim/config.py` | 설정 dataclass 와 입력 검증 |
| `polarsim/physics.py` | 엔진 A: 평면파/가우시안 빔 해석적 중첩, 수치 회전 검증 |
| `polarsim/metrics.py` | 시간창, 궤적 지표 (L, e, 평면성, T), 실행 |
| `polarsim/presets.py` | 프리셋 3종과 예상 결과 |
| `polarsim/fdtd.py` | 엔진 B: 2D 벽·반사·회절 (주파수 영역 정상상태 + FDTD 교차검증), 임의 각도 막대 프레임 |
| `polarsim/gui/frame_editor.py` | 엔진 B 로봇 프레임(막대) 편집기 |
| `polarsim/engine3d/` | 3D 소나: 설정·예시·JSON (`config3d`), 형상·STL 격자화 (`geometry3d`), 3D FDTD (`fdtd3d`), 검증용 해석해 (`analytic3d`) |
| `polarsim/gui/tab3d.py`, `view3d.py` | 3D 소나 탭과 CAD 식 3D 뷰 (pyqtgraph.opengl) |
| `polarsim/display.py` | 화면 뭉침 정도 (면적비) 와 시각 배율 계산 |
| `polarsim/export.py` | CSV / JSON / 요약 저장 |
| `polarsim/gui/` | PySide6 + pyqtgraph GUI |
| `polarsim/app.py` | 진입점, 로그, 예외 대화상자, `--selftest` |
| `tests/` | pytest (물리 모듈은 GUI 없이 실행) |
| `analysis/pocket_radiation.py` | 품 밖 타원도 vs 거리 분석 (`python -m analysis.pocket_radiation`) |
| `build.ps1`, `polarsim.spec`, `.github/workflows/build-windows.yml` | Windows exe 빌드 |
