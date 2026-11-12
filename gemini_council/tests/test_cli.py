"""명령줄 진입점 테스트."""
import sys
from pathlib import Path

import pytest

from council.cli import main, split_command

FAKE = Path(__file__).with_name("fake_gemini.py")


@pytest.mark.parametrize("s,windows,expected", [
    ("gemini", False, ("gemini",)),
    ("npx @google/gemini-cli", False, ("npx", "@google/gemini-cli")),
    (r"C:\Users\연구실\AppData\Roaming\npm\gemini.cmd", True,
     (r"C:\Users\연구실\AppData\Roaming\npm\gemini.cmd",)),          # 백슬래시 보존
    (r'"C:\Program Files\nodejs\npx.cmd" @google/gemini-cli', True,
     (r"C:\Program Files\nodejs\npx.cmd", "@google/gemini-cli")),   # 공백 경로 + 따옴표
])
def test_split_command(s, windows, expected):
    assert split_command(s, windows=windows) == expected


def test_existing_path_with_space_kept_whole(tmp_path):
    exe = tmp_path / "my tools" / "gemini"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    assert split_command(str(exe), windows=False) == (str(exe),)


def _fake_cmd():
    return f'"{sys.executable}" "{FAKE}"' if sys.platform == "win32" else f"{sys.executable} {FAKE}"


def test_cli_check_ok(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FAKE_GEMINI_MODE", "ok")
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(tmp_path / "c.jsonl"))
    assert main(["--backend", "gemini-cli", "--gemini-cmd", _fake_cmd(), "--check", "--min-interval", "0"]) == 0
    assert "합계 (2회)" in capsys.readouterr().out


def test_cli_check_auth_fail(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FAKE_GEMINI_MODE", "auth")
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(tmp_path / "c.jsonl"))
    assert main(["--backend", "gemini-cli", "--gemini-cmd", _fake_cmd(), "--check"]) == 1
    assert "로그인이 필요함" in capsys.readouterr().out


def test_cli_quota_returns_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_GEMINI_MODE", "quota")
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(tmp_path / "c.jsonl"))
    rc = main(["--backend", "gemini-cli", "--gemini-cmd", _fake_cmd(), "--topic", "t",
               "--log-dir", str(tmp_path / "logs")])
    assert rc == 2 and len(list((tmp_path / "logs").glob("*.md"))) == 1


def test_cli_requires_topic():
    with pytest.raises(SystemExit):
        main(["--backend", "mock"])


def test_ask_interactive_defaults_and_validation(capsys):
    from council.cli import INTERACTIVE_DEFAULT_ROUNDS, ask_interactive
    answers = iter(["", "  로봇 팔 제어  ", "", "abc", "0", ""])
    argv = ask_interactive(read=lambda prompt: next(answers))
    assert argv == ["--backend", "manual", "--topic", "로봇 팔 제어", "--rounds", str(INTERACTIVE_DEFAULT_ROUNDS),
                    "--agenda", "general"]
    assert capsys.readouterr().out.count("1 이상의 숫자") == 2
    answers = iter(["주제", "9", "2", "5"])  # 잘못된 유형 9 → 다시 물음 → 2(주제 검토)
    argv = ask_interactive(read=lambda prompt: next(answers))
    assert argv[-4:] == ["--rounds", "5", "--agenda", "topic-review"]


def test_default_backend_is_manual():
    from council.cli import build_parser
    assert build_parser().parse_args(["--topic", "t"]).backend == "manual"
