<#
.SYNOPSIS
  gemini_council Windows 빌드: 가상환경 -> pytest -> PyInstaller(단일 exe) -> exe 단독 실행 점검.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build.ps1
  powershell -ExecutionPolicy Bypass -File .\build.ps1 -SkipTests
.NOTES
  결과: dist\gemini_council.exe (Python 없는 PC 에서도 실행. 단, Gemini CLI 와 Node.js 는 따로 필요)
#>
param(
    [switch]$SkipTests,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
# exe/파이썬이 UTF-8 로 출력한 한글을 PowerShell 이 올바르게 읽도록
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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
Invoke-Checked "의존성 설치" { & $py -m pip install --quiet -r requirements-dev.txt "pyinstaller>=6.10" }

if (-not $SkipTests) {
    Invoke-Checked "pytest" { & $py -m pytest -q }
}

Invoke-Checked "PyInstaller 빌드 (단일 exe)" {
    & $py -m PyInstaller --noconfirm --clean --onefile --console --name gemini_council `
        --distpath dist --workpath build --specpath build run_council.py
}
$exe = Join-Path $PSScriptRoot "dist\gemini_council.exe"

# ---- exe 단독 실행 점검: 소스 트리와 분리된 임시 폴더에서 실행 ----
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$clean = Join-Path ([System.IO.Path]::GetTempPath()) "gemini_council_exe_$stamp"
New-Item -ItemType Directory -Path $clean | Out-Null
Copy-Item $exe $clean
$cleanExe = Join-Path $clean "gemini_council.exe"
Push-Location $clean
try {
    Invoke-Checked "exe --help" { & $cleanExe --help | Out-Null }
    Invoke-Checked "exe mock 토론 (1라운드)" {
        & $cleanExe --backend mock --topic "exe 점검 주제" --rounds 2 --mock-end-round 1 --log-dir "로그 폴더"
    }
    $md = Get-ChildItem -Path (Join-Path $clean "로그 폴더") -Filter "discussion_*.md"
    if ($md.Count -ne 1) { throw "회의록이 1개가 아님: $($md.Count)" }
    $text = Get-Content $md[0].FullName -Raw -Encoding UTF8
    if ($text -notmatch "사회자 종료 선언") { throw "회의록 내용 이상" }
    Write-Host "회의록 확인: $($md[0].FullName)" -ForegroundColor Green
    # 반자동 모드의 핵심인 Windows 클립보드(ctypes)가 exe 안에서도 동작하는지
    Invoke-Checked "exe 클립보드 점검" { & $cleanExe --clipboard-test }

    # Gemini CLI 가 설치되어 있으면 실제 연결 확인 (로그인 안 됐으면 exit 1 이 정상)
    if (Get-Command gemini.cmd -ErrorAction SilentlyContinue) {
        Write-Host "==> exe --backend gemini-cli --check" -ForegroundColor Cyan
        & $cleanExe --backend gemini-cli --check
        Write-Host "  (exit ${LASTEXITCODE}: 0=로그인·모델 정상, 1=로그인 필요 또는 실패 원인 위 메시지 참고)"
    } else {
        Write-Host "Gemini CLI 미설치: 실제 연결 점검 생략" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}
Write-Host "빌드 완료: $exe" -ForegroundColor Green
# 위 --check 의 exit 1(로그인 필요)이 스크립트 종료 코드로 새지 않게 한다
exit 0
