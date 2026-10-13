# polarsim — 다중 음파 중첩에 의한 입자 운동 편광 시뮬레이터

서로 다른 방향의 음파(종파)를 중첩했을 때 공기 입자 궤적이 직선에서 타원/원으로 바뀌는 정도를 정량화한다.
횡파가 생기는 것이 아니며 변위장은 항상 비회전(∇×u = 0)이다. 자세한 내용은 [사용자 가이드](docs/USER_GUIDE_ko.md).

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
| `polarsim/export.py` | CSV / JSON / 요약 저장 |
| `polarsim/gui/` | PySide6 + pyqtgraph GUI |
| `polarsim/app.py` | 진입점, 로그, 예외 대화상자, `--selftest` |
| `tests/` | pytest (물리 모듈은 GUI 없이 실행) |
| `build.ps1`, `polarsim.spec`, `.github/workflows/build-windows.yml` | Windows exe 빌드 |
