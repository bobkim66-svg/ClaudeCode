<#
.SYNOPSIS
  polarsim Windows 빌드: 가상환경 -> pytest -> PyInstaller -> clean 폴더에서 exe 단독 자체진단.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build.ps1             # 기본: 폴더(onedir) 빌드
  powershell -ExecutionPolicy Bypass -File .\build.ps1 -OneFile    # 단일 exe (옵션)
  powershell -ExecutionPolicy Bypass -File .\build.ps1 -SkipTests
.NOTES
  결과: dist\polarsim\polarsim.exe (onedir) 또는 dist\polarsim.exe (onefile).
  dist\ 와 build\ 는 매 빌드마다 새로 만들어진다 (빌드 산출물). 자체진단 결과는 selftest_out\<시각>\ 에 쌓인다.
#>
param(
    [switch]$OneFile,
    [switch]$SkipTests,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Invoke-Checked([string]$What, [scriptblock]$Block) {
    Write-Host "==> $What" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What 실패 (exit $LASTEXITCODE)" }
}

$venv = Join-Path $PSScriptRoot ".venv-build"
if (-not (Test-Path $venv)) {
    Invoke-Checked "가상환경 생성 ($Python)" { & $Python -m venv $venv }
}
$py = Join-Path $venv "Scripts\python.exe"
Invoke-Checked "Python 버전 확인" { & $py --version }
Invoke-Checked "pip 업그레이드" { & $py -m pip install --upgrade pip }
Invoke-Checked "의존성 설치 (실패 시: Python 버전에 맞는 휠이 있는지 확인)" { & $py -m pip install -r requirements-dev.txt }

if (-not $SkipTests) {
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Checked "pytest" { & $py -m pytest -q }
    Remove-Item Env:\QT_QPA_PLATFORM
}

$env:POLARSIM_ONEFILE = if ($OneFile) { "1" } else { "0" }
Invoke-Checked "PyInstaller 빌드" { & $py -m PyInstaller --noconfirm --clean polarsim.spec }

# ---- clean 폴더에서 단독 실행 -------------------------------------------------
# 소스 트리/가상환경과 분리된 임시 폴더로 복사하고, PATH 에서 Python 을 뺀 상태로 실행한다.
# (Python 미설치 PC 의 근사. 시스템에 설치된 DLL 까지 없앨 수는 없으므로 완전한 대체는 아님)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$clean = Join-Path ([System.IO.Path]::GetTempPath()) "polarsim_clean_$stamp"
New-Item -ItemType Directory -Path $clean | Out-Null
if ($OneFile) { Copy-Item "dist\polarsim.exe" $clean } else { Copy-Item "dist\polarsim" $clean -Recurse }
$cleanExe = (Get-ChildItem -Path $clean -Recurse -Filter "polarsim.exe" | Select-Object -First 1).FullName
$report = Join-Path $PSScriptRoot "selftest_out\$stamp"

$savedPath = $env:PATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
Remove-Item Env:\PYTHONPATH, Env:\PYTHONHOME -ErrorAction SilentlyContinue
try {
    Write-Host "==> exe 자체진단: $cleanExe --selftest $report" -ForegroundColor Cyan
    # CI 러너(GPU 없음)는 OpenGL 2.0 이 없어 기본 GL 로는 3D 화면을 그릴 수 없다 (실측: glClearColor GLError 1282).
    # 그래서 GPU 없는 PC 용 대체 경로인 소프트웨어 OpenGL 로 3D 렌더링을 검증한다.
    $proc = Start-Process -FilePath $cleanExe -ArgumentList @("--selftest", "`"$report`"", "--software-gl") -PassThru
    $timedOut = -not $proc.WaitForExit(900000)  # 15 분: 멈추면 (예: 대화상자) 무한 대기하지 않게
    if ($timedOut) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
} finally {
    $env:PATH = $savedPath
}
# 진단 근거를 CI 로그에 바로 남긴다 (아티팩트를 받지 않아도 원인 확인 가능)
$appLog = Join-Path $env:LOCALAPPDATA "polarsim\logs\polarsim.log"
if (Test-Path $appLog) {
    Write-Host "---- polarsim.log (마지막 60줄) ----" -ForegroundColor Cyan
    Get-Content -Path $appLog -Encoding UTF8 -Tail 60 | Write-Host
}
$hang = Join-Path $report "hang_traceback.txt"
if ((Test-Path $hang) -and (Get-Item $hang).Length -gt 0) {
    Write-Host "---- hang_traceback.txt ----" -ForegroundColor Cyan
    Get-Content -Path $hang -Encoding UTF8 | Write-Host
}
if ($timedOut) { throw "exe 자체진단 시간 초과 (15 분). 위 로그의 마지막 단계에서 멈췄습니다." }

$reportFile = Join-Path $report "selftest_report.txt"
if (Test-Path $reportFile) {
    Get-Content -Path $reportFile -Encoding UTF8 | Write-Host
} else {
    Write-Host "자체진단 보고서가 없습니다. 로그를 확인하세요: $env:LOCALAPPDATA\polarsim\logs\polarsim.log" -ForegroundColor Red
}
if ($proc.ExitCode -ne 0) { throw "exe 자체진단 실패 (exit $($proc.ExitCode))" }
Write-Host "빌드 완료: $(Resolve-Path ($(if ($OneFile) { 'dist\polarsim.exe' } else { 'dist\polarsim' })))" -ForegroundColor Green
