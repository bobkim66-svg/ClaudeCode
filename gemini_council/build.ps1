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
        --collect-all playwright `
        --distpath dist --workpath build --specpath build run_council.py
}
Invoke-Checked "PyInstaller 빌드 (Deep Research 대기열 exe)" {
    & $py -m PyInstaller --noconfirm --onefile --console --name gemini_research `
        --collect-all playwright `
        --distpath dist --workpath build --specpath build run_research.py
}
$researchExe = Join-Path $PSScriptRoot "dist\gemini_research.exe"
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

    # 자동(web) 모드: exe 에 묶인 Playwright 로 실제 Chrome(헤드리스)을 띄워 가짜 Gemini 페이지와 끝까지 토론
    $fakePage = ([System.Uri](Join-Path $PSScriptRoot "tests\fake_gemini_page.html")).AbsoluteUri
    $webProfile = Join-Path $clean "web_profile"
    New-Item -ItemType Directory -Path $webProfile | Out-Null   # 로그인 단계 건너뛰기 (가짜 페이지)
    try {
        Invoke-Checked "exe 자동(web) 모드 점검" {
            & $cleanExe --backend web --web-url "$fakePage`?end=1" --chrome-arg=--headless=new `
                --profile-dir $webProfile --port 9411 --topic "exe 웹 점검" --rounds 2 --interval 0 --log-dir "웹 로그"
        }
        $wmd = Get-ChildItem -Path (Join-Path $clean "웹 로그") -Filter "discussion_*.md"
        $wtext = Get-Content $wmd[0].FullName -Raw -Encoding UTF8
        if ($wtext -notmatch "사회자 종료 선언" -or $wtext -notmatch "가짜 초록") { throw "자동 모드 회의록 내용 이상" }
        Write-Host "자동 모드 회의록 확인: $($wmd[0].FullName)" -ForegroundColor Green
    } finally {
        Get-CimInstance Win32_Process -Filter "Name = 'chrome.exe'" |
            Where-Object { $_.CommandLine -like "*$webProfile*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }

    # Deep Research 대기열 exe: 가짜 Deep Research 페이지로 주제 2개를 끝까지 (모드 선택, 도구 메뉴, '연구 시작')
    Copy-Item $researchExe $clean
    $cleanResearch = Join-Path $clean "gemini_research.exe"
    $fakeResearch = ([System.Uri](Join-Path $PSScriptRoot "tests\fake_research_page.html")).AbsoluteUri
    "# 시험`n일반 | 열화상 깊이 추정`n사고 | 레이더 내비게이션`n" | Out-File -FilePath (Join-Path $clean "research_topics.txt") -Encoding utf8
    try {
        Invoke-Checked "Deep Research exe 점검" {
            & $cleanResearch --web-url $fakeResearch --chrome-arg=--headless=new --profile-dir $webProfile `
                --port 9412 --interval 0 --stable-sec 1
        }
        $reports = Get-ChildItem -Path (Join-Path $clean "research") -Filter "*.md"
        if ($reports.Count -ne 2) { throw "보고서가 2개가 아님: $($reports.Count)" }
        foreach ($r in $reports) {
            if ((Get-Content $r.FullName -Raw -Encoding UTF8) -notmatch "가짜 보고서 끝") { throw "보고서 내용 이상: $($r.Name)" }
        }
        Write-Host "Deep Research 보고서 확인: $($reports.Name -join ', ')" -ForegroundColor Green
    } finally {
        Get-CimInstance Win32_Process -Filter "Name = 'chrome.exe'" |
            Where-Object { $_.CommandLine -like "*$webProfile*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }

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
