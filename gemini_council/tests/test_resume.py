"""자동 저장과 이어 하기 테스트."""
import json
from datetime import datetime, timedelta
import pytest

from council.backends import MockBackend
from council.backends.manual import RESET_PREFIX, ManualRelayBackend
from council.cli import ask_interactive, main
from council.config import CouncilConfig
from council.orchestrator import Orchestrator, StopReason
from council.protocol import RECAP_NOTE
from council.roles import DEFAULT_ROLES
from council.session import SessionError, find_unfinished, load_state, write_atomic

TOPIC = "이어 하기 시험 주제"
N = len(DEFAULT_ROLES)


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 12, 9, 0, 0)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


def _interrupted_run(tmp_path, fail_at=6, rounds=3):
    """fail_at 번째 발언에서 사용량 한도로 중단된 토론을 만든다 → (결과, md 경로)."""
    cfg = CouncilConfig(max_rounds=rounds, log_dir=tmp_path / "logs")
    res = Orchestrator(MockBackend(fail_at_turn=fail_at, fail_with="limit"), TOPIC, cfg, clock=Clock()).run()
    return res, res.log_path


# ---------- 자동 저장 ----------

def test_autosave_after_every_utterance(tmp_path):
    seen = []
    cfg = CouncilConfig(max_rounds=2, log_dir=tmp_path / "logs")

    def on_event(msg):
        if "응답" in msg and "[R" in msg:  # 각 발언 저장 직후 상태 파일을 읽어 본다
            js = next((tmp_path / "logs").glob("*.json"))
            d = json.loads(js.read_text(encoding="utf-8"))
            seen.append((d["status"], len(d["utterances"])))

    res = Orchestrator(MockBackend(), TOPIC, cfg, on_event=on_event, clock=Clock()).run()
    assert seen == [("RUNNING", k) for k in range(1, 2 * N + 1)]
    final = json.loads(res.log_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert final["status"] == "MAX_ROUNDS" and len(final["utterances"]) == 2 * N
    assert len(list((tmp_path / "logs").glob("*.md"))) == 1   # 같은 파일을 갱신 (새 파일을 계속 만들지 않음)
    assert not list((tmp_path / "logs").glob("*.tmp"))         # 원자적 교체 후 임시 파일 없음


def test_interrupted_transcript_has_resume_hint(tmp_path):
    res, md = _interrupted_run(tmp_path)
    assert res.reason is StopReason.USAGE_LIMIT and len(res.utterances) == 5
    text = md.read_text(encoding="utf-8")
    assert "## 이어서 하려면" in text and "--resume" in text
    done = Orchestrator(MockBackend(end_at_round=1), TOPIC, CouncilConfig(max_rounds=2, log_dir=tmp_path / "x"),
                        clock=Clock()).run()
    assert "## 이어서 하려면" not in done.log_path.read_text(encoding="utf-8")  # 정상 종료면 안내 없음


def test_locked_file_during_autosave_warns_and_continues(tmp_path, monkeypatch):
    import council.orchestrator as orch_mod
    real = orch_mod.write_atomic
    calls = {"n": 0}

    def flaky(path, text):
        calls["n"] += 1
        if 3 <= calls["n"] <= 5:   # 2번째 발언의 자동 저장 3회 시도 모두 잠김
            raise PermissionError("파일이 다른 프로세스에서 사용 중")
        real(path, text)

    monkeypatch.setattr(orch_mod, "write_atomic", flaky)
    monkeypatch.setattr(orch_mod.Orchestrator, "SAVE_RETRY_S", 0.0)
    events = []
    res = Orchestrator(MockBackend(), TOPIC, CouncilConfig(max_rounds=1, log_dir=tmp_path / "logs"),
                       on_event=events.append, clock=Clock()).run()
    assert res.reason is StopReason.MAX_ROUNDS and len(res.utterances) == N    # 토론은 끝까지 진행
    assert sum("자동 저장 실패" in e for e in events) == 1
    assert len(json.loads(res.log_path.with_suffix(".json").read_text(encoding="utf-8"))["utterances"]) == N


# ---------- 이어 하기 ----------

def test_resume_continues_from_next_turn_with_recap(tmp_path):
    _, md = _interrupted_run(tmp_path, fail_at=6, rounds=3)   # 발언 5개 = R1 4개 + R2 제안자
    data, js = load_state(md)
    be = MockBackend(seed=1)
    res = Orchestrator(be, data["topic"], CouncilConfig(max_rounds=3, log_dir=js.parent),
                       resume=data, resume_json=js, clock=Clock()).run()
    assert res.reason is StopReason.MAX_ROUNDS and len(res.utterances) == 3 * N
    assert res.log_path == md                                    # 같은 회의록에 이어서 기록
    keys = [k for k, _ in be.prompts]
    assert keys[0] == "critic"                                   # 6번째 차례 = R2 비평가부터
    first = {}
    for k, p in be.prompts:
        first.setdefault(k, []).append(p)
    for k, ps in first.items():
        assert RECAP_NOTE in ps[0] and "[토론 규칙]" in ps[0]    # 이어 하기 첫 메시지: 규칙 + 맥락 복구
        assert all(RECAP_NOTE not in p and "[토론 규칙]" not in p for p in ps[1:])  # 이후는 평소대로 새 발언만
    final = json.loads(js.read_text(encoding="utf-8"))
    assert len(final["resumes"]) == 1 and final["status"] == "MAX_ROUNDS"
    assert "중단 후 이어 하기: 1회" in md.read_text(encoding="utf-8")
    for k, u in enumerate(res.utterances):                       # 순서 불변 조건
        assert u.speaker_key == DEFAULT_ROLES[k % N].key and u.round == k // N + 1


def test_recap_contains_summary_recent_and_own_last(tmp_path):
    _, md = _interrupted_run(tmp_path, fail_at=7)                 # R1 4개 + R2 제안자, 비평가
    data, js = load_state(md)
    utts = data["utterances"]
    be = MockBackend()
    Orchestrator(be, TOPIC, CouncilConfig(max_rounds=2, log_dir=js.parent), resume=data, resume_json=js,
                 clock=Clock()).run()
    p = be.prompts[0][1]                                          # R2 대안 설계자의 이어 하기 첫 메시지
    assert utts[3]["text"].strip() in p                           # 사회자 최신 요약 (R1)
    assert utts[4]["text"].strip() in p and utts[5]["text"].strip() in p   # 요약 이후 발언 (R2 제안자·비평가)
    assert utts[2]["text"].strip() in p                           # 자기 직전 발언 (R1 대안 설계자)
    assert utts[0]["text"].strip() not in p                       # 요약 이전의 다른 발언은 요약으로 대체


def test_resume_refusals(tmp_path):
    res = Orchestrator(MockBackend(end_at_round=1), TOPIC, CouncilConfig(max_rounds=3, log_dir=tmp_path / "a"),
                       clock=Clock()).run()
    data, js = load_state(res.log_path)
    with pytest.raises(SessionError, match="종료를 선언"):
        Orchestrator(MockBackend(), TOPIC, CouncilConfig(max_rounds=3, log_dir=js.parent), resume=data,
                     resume_json=js)
    res = Orchestrator(MockBackend(), TOPIC, CouncilConfig(max_rounds=1, log_dir=tmp_path / "b"),
                       clock=Clock()).run()
    data, js = load_state(res.log_path)
    with pytest.raises(SessionError, match="최대 라운드"):
        Orchestrator(MockBackend(), TOPIC, CouncilConfig(max_rounds=1, log_dir=js.parent), resume=data,
                     resume_json=js)
    more = Orchestrator(MockBackend(), TOPIC, CouncilConfig(max_rounds=2, log_dir=js.parent), resume=data,
                        resume_json=js, clock=Clock()).run()          # 라운드를 늘리면 이어 할 수 있음
    assert len(more.utterances) == 2 * N
    _, md = _interrupted_run(tmp_path / "c")
    data, js = load_state(md)
    data["utterances"][1]["speaker_key"] = "proposer"                 # 손상된 순서
    with pytest.raises(SessionError, match="순서"):
        Orchestrator(MockBackend(), TOPIC, CouncilConfig(log_dir=js.parent), resume=data, resume_json=js)


def test_load_state_errors(tmp_path):
    with pytest.raises(SessionError, match="없음"):
        load_state(tmp_path / "nope.md")
    bad = tmp_path / "discussion_x.json"
    bad.write_text("{ broken", encoding="utf-8")
    with pytest.raises(SessionError, match="손상"):
        load_state(bad)


def test_find_unfinished_picks_latest_interrupted(tmp_path):
    import os
    logs = tmp_path / "logs"
    assert find_unfinished(logs) is None
    _, md1 = _interrupted_run(tmp_path)
    done = Orchestrator(MockBackend(end_at_round=1), TOPIC, CouncilConfig(log_dir=logs), clock=Clock()).run()
    (logs / "discussion_broken.json").write_text("{", encoding="utf-8")
    later = md1.with_suffix(".json").stat().st_mtime + 100
    os.utime(done.log_path.with_suffix(".json"), (later, later))      # 완료된 것이 더 최근이어도
    assert find_unfinished(logs) == md1.with_suffix(".json")         # 중단된 것을 고른다


def test_write_atomic_replaces_whole_file(tmp_path):
    f = tmp_path / "a.md"
    write_atomic(f, "첫 내용 " * 1000)
    write_atomic(f, "새 내용")
    assert f.read_text(encoding="utf-8") == "새 내용" and not list(tmp_path.glob("*.tmp"))


# ---------- 반자동: 재활용 탭 리셋 ----------

def test_manual_reset_prefix_only_on_first_message_per_tab(tmp_path):
    from tests.test_manual import FakeClipboard, TabSimulator  # 같은 테스트 도구 재사용
    clip = FakeClipboard()
    sim = TabSimulator(clip)
    be = ManualRelayBackend(clipboard=clip, out=lambda s: None, sleep=sim.sleep, clock=lambda: sim.t)
    Orchestrator(be, TOPIC, CouncilConfig(max_rounds=2, log_dir=tmp_path)).run()
    for key, inputs in sim.tab_inputs.items():
        assert inputs[0].startswith(RESET_PREFIX) and not inputs[1].startswith(RESET_PREFIX)


# ---------- CLI ----------

def test_cli_resume_end_to_end(tmp_path, capsys):
    _, md = _interrupted_run(tmp_path, rounds=2)
    rc = main(["--backend", "mock", "--resume", str(md)])
    out = capsys.readouterr().out
    assert rc == 0 and "이어 하기: 발언 5개 다음부터" in out
    assert len(json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))["utterances"]) == 2 * N
    assert main(["--backend", "mock", "--resume", str(tmp_path / "없음.md")]) == 1
    assert "이어 하기 실패" in capsys.readouterr().out


def test_interactive_offers_resume(tmp_path, capsys):
    _, md = _interrupted_run(tmp_path)
    answers = iter(["y", "", ""])                                     # 이어 하기 → 방식 Enter(반자동) → 간격 Enter
    argv = ask_interactive(read=lambda q: next(answers), log_dir=md.parent)
    assert argv == ["--backend", "manual", "--resume", str(md.with_suffix(".json")), "--interval", "5"]
    assert TOPIC in capsys.readouterr().out
    answers = iter(["", "", "새 주제", "", "", ""])                   # Enter = 새 토론, 방식 Enter
    argv = ask_interactive(read=lambda q: next(answers), log_dir=md.parent)
    assert "--resume" not in argv and argv[argv.index("--topic") + 1] == "새 주제"
