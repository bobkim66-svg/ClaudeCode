"""오케스트레이터 + mock 백엔드 통합 테스트: 모든 종료 경로에서 회의록이 남는지."""
import threading
from datetime import datetime

import pytest

from council.backends import MockBackend
from council.config import CouncilConfig
from council.orchestrator import Orchestrator, StopReason
from council.roles import DEFAULT_ROLES
from council.transcript import save_transcript

TOPIC = "수중 로봇 경로 계획에서 음파 측위 오차 처리"
N_ROLES = len(DEFAULT_ROLES)
FIXED = datetime(2026, 11, 12, 9, 0, 0)


def _run(tmp_path, rounds=5, **mock_kw):
    cfg = CouncilConfig(max_rounds=rounds, log_dir=tmp_path / "logs")
    be = MockBackend(end_token=cfg.end_token, **mock_kw)
    res = Orchestrator(be, TOPIC, cfg, clock=lambda: FIXED).run()
    return res, be


def _assert_turn_order(res):
    """불변 조건: k번째 발언자는 역할 순서의 (k mod 4)번째, 라운드 번호는 k//4+1."""
    for k, u in enumerate(res.utterances):
        assert u.speaker_key == DEFAULT_ROLES[k % N_ROLES].key
        assert u.round == k // N_ROLES + 1


def test_moderator_ends_discussion(tmp_path):
    res, be = _run(tmp_path, end_at_round=3)
    assert res.reason is StopReason.COMPLETED
    assert res.rounds_done == 3 and len(res.utterances) == 3 * N_ROLES
    assert be.opened and be.closed
    _assert_turn_order(res)
    md = res.log_path.read_text(encoding="utf-8")
    assert all(u.text.strip() in md for u in res.utterances)  # 모든 발언 누락 없이 저장
    assert "사회자 종료 선언" in md


def test_max_rounds(tmp_path):
    res, _ = _run(tmp_path, rounds=4)
    assert res.reason is StopReason.MAX_ROUNDS
    assert res.rounds_done == 4 and len(res.utterances) == 4 * N_ROLES
    _assert_turn_order(res)


def test_single_round_edge(tmp_path):
    res, _ = _run(tmp_path, rounds=1)
    assert res.reason is StopReason.MAX_ROUNDS and len(res.utterances) == N_ROLES


@pytest.mark.parametrize("kind,reason", [
    ("limit", StopReason.USAGE_LIMIT),
    ("timeout", StopReason.TIMEOUT),
    ("error", StopReason.BACKEND_ERROR),
])
def test_backend_failure_saves_partial(tmp_path, kind, reason):
    res, be = _run(tmp_path, fail_at_turn=6, fail_with=kind)
    assert res.reason is reason
    assert len(res.utterances) == 5      # 6번째 턴에서 실패 → 5개 저장
    assert res.rounds_done == 1          # 라운드 2는 미완료
    assert be.closed
    assert res.log_path.exists()
    assert reason.value in res.log_path.read_text(encoding="utf-8")


def test_unexpected_error_is_saved_and_reraised(tmp_path):
    cfg = CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")
    be = MockBackend(fail_at_turn=3, fail_with="crash")
    with pytest.raises(RuntimeError, match="모의 예기치 못한 오류"):
        Orchestrator(be, TOPIC, cfg).run()
    files = list((tmp_path / "logs").glob("discussion_*.md"))
    assert len(files) == 1
    assert "예기치 못한 오류" in files[0].read_text(encoding="utf-8")


def test_user_stop_before_start(tmp_path):
    cfg = CouncilConfig(max_rounds=3, log_dir=tmp_path / "logs")
    stop = threading.Event()
    stop.set()
    res = Orchestrator(MockBackend(), TOPIC, cfg, stop_event=stop).run()
    assert res.reason is StopReason.USER_STOP and res.utterances == ()


def test_user_stop_mid_run(tmp_path):
    cfg = CouncilConfig(max_rounds=5, log_dir=tmp_path / "logs")
    stop = threading.Event()
    msgs = []

    def on_event(msg):
        msgs.append(msg)
        if msg.startswith("[R2] 제안자 응답"):
            stop.set()

    res = Orchestrator(MockBackend(), TOPIC, cfg, on_event=on_event, stop_event=stop).run()
    assert res.reason is StopReason.USER_STOP
    assert len(res.utterances) == N_ROLES + 1


def test_pause_then_resume(tmp_path):
    """일시정지 중에는 발언이 진행되지 않고, 재개하면 끝까지 진행된다."""
    cfg = CouncilConfig(max_rounds=2, log_dir=tmp_path / "logs")
    resume = threading.Event()  # clear = 일시정지 상태로 시작
    orch = Orchestrator(MockBackend(), TOPIC, cfg, resume_event=resume)
    t = threading.Thread(target=orch.run)
    t.start()
    t.join(timeout=0.5)
    assert t.is_alive() and orch.state.utterances == []
    resume.set()
    t.join(timeout=5)
    assert not t.is_alive() and len(orch.state.utterances) == 2 * N_ROLES


def test_reproducible_with_same_seed(tmp_path):
    a, _ = _run(tmp_path / "a", seed=42)
    b, _ = _run(tmp_path / "b", seed=42)
    c, _ = _run(tmp_path / "c", seed=7)
    assert a.log_path.read_text(encoding="utf-8") == b.log_path.read_text(encoding="utf-8")
    assert [u.text for u in a.utterances] != [u.text for u in c.utterances]


def test_stateless_mode_runs(tmp_path):
    res, be = _run(tmp_path, rounds=3, stateful=False)
    assert res.reason is StopReason.MAX_ROUNDS
    turns = [p for _, p in be.prompts if "초록 작성" not in p]
    assert len(turns) == 3 * N_ROLES and all("[토론 규칙]" in p for p in turns)  # stateless 는 매 턴 seed 포함
    abstract = [p for _, p in be.prompts if "초록 작성" in p]
    assert len(abstract) == 1 and "[토론 전체 발언]" in abstract[0]      # stateless 초록 요청엔 전체 발언 동봉
    assert all(u.text.strip() in abstract[0] for u in res.utterances)


def test_stateful_mode_sends_seed_once_per_role(tmp_path):
    _, be = _run(tmp_path, rounds=3)
    seeds = [k for k, p in be.prompts if "[토론 규칙]" in p]
    assert sorted(seeds) == sorted(r.key for r in DEFAULT_ROLES)


def test_empty_topic_rejected(tmp_path):
    with pytest.raises(ValueError):
        Orchestrator(MockBackend(), "   ", CouncilConfig(log_dir=tmp_path))


def test_transcript_no_overwrite_and_utf8(tmp_path):
    p1 = save_transcript("# 한글 회의록 ①", tmp_path / "새 폴더", FIXED)
    p2 = save_transcript("# 두 번째", tmp_path / "새 폴더", FIXED)
    assert p1 != p2 and p2.name == "discussion_20261112_090000_1.md"
    assert p1.read_text(encoding="utf-8") == "# 한글 회의록 ①"
