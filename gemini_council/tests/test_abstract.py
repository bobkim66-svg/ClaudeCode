"""토론 종료 후 초록(전체 요약) 테스트."""
import json

from council.backends import MockBackend, UserAbort
from council.cli import main
from council.config import CouncilConfig
from council.orchestrator import Orchestrator, StopReason
from council.protocol import ABSTRACT_MARKER, DiscussionState, Utterance, build_abstract_prompt
from council.roles import AGENDAS, DEFAULT_ROLES

TOPIC = "초록 시험 주제"


class AbortOnAbstract(MockBackend):
    def send(self, role, prompt):
        if ABSTRACT_MARKER in prompt:
            raise UserAbort("q")
        return super().send(role, prompt)


def test_abstract_on_completion_and_max_rounds(tmp_path):
    for kw, reason in (({"end_at_round": 1}, StopReason.COMPLETED), ({}, StopReason.MAX_ROUNDS)):
        be = MockBackend(**kw)
        res = Orchestrator(be, TOPIC, CouncilConfig(max_rounds=2, log_dir=tmp_path / reason.name)).run()
        assert res.reason is reason
        md = res.log_path.read_text(encoding="utf-8")
        assert "## 초록" in md and "모의 결론" in md and md.index("## 초록") < md.index("## 라운드 1")
        assert json.loads(res.log_path.with_suffix(".json").read_text(encoding="utf-8"))["abstract"]
        assert sum(ABSTRACT_MARKER in p for _, p in be.prompts) == 1
        assert be.prompts[-1][0] == "moderator"                  # 사회자에게 요청


def test_no_abstract_when_interrupted_or_disabled(tmp_path):
    be = MockBackend(fail_at_turn=3, fail_with="limit")
    res = Orchestrator(be, TOPIC, CouncilConfig(max_rounds=2, log_dir=tmp_path / "a")).run()
    assert res.reason is StopReason.USAGE_LIMIT and not any(ABSTRACT_MARKER in p for _, p in be.prompts)
    assert "## 초록" not in res.log_path.read_text(encoding="utf-8")
    be = MockBackend()
    res = Orchestrator(be, TOPIC, CouncilConfig(max_rounds=1, log_dir=tmp_path / "b"), make_abstract=False).run()
    assert not any(ABSTRACT_MARKER in p for _, p in be.prompts)


def test_quit_during_abstract_keeps_completed_discussion(tmp_path):
    res = Orchestrator(AbortOnAbstract(end_at_round=1), TOPIC, CouncilConfig(max_rounds=2, log_dir=tmp_path)).run()
    assert res.reason is StopReason.COMPLETED and len(res.utterances) == 4   # 결과는 그대로
    md = res.log_path.read_text(encoding="utf-8")
    assert "초록 단계에서 사용자가 중단해 초록 없음" in md and "## 이어서 하려면" not in md


def test_abstract_prompt_contents():
    s = DiscussionState()
    for r in DEFAULT_ROLES:
        s.record(Utterance(1, r.key, r.name, f"{r.name} 발언"))
    cfg = CouncilConfig(agenda=AGENDAS["topic-review"])
    stateful = build_abstract_prompt(DEFAULT_ROLES, s, TOPIC, cfg, stateful=True)
    assert ABSTRACT_MARKER in stateful and "제안자 발언" not in stateful     # 사회자는 이미 다 봄
    assert "점검 항목별 최종 상태" in stateful and "새 주장을 추가하지 않는다" in stateful
    s.record(Utterance(2, "proposer", "제안자", "사회자가 못 본 발언"))
    assert "사회자가 못 본 발언" in build_abstract_prompt(DEFAULT_ROLES, s, TOPIC, cfg, stateful=True)
    stateless = build_abstract_prompt(DEFAULT_ROLES, s, TOPIC, CouncilConfig(), stateful=False)
    assert all(u.text in stateless for u in s.utterances) and "점검 항목" not in stateless
    assert "[DISCUSSION_COMPLETE]" not in stateless


def test_cli_no_abstract_flag(tmp_path):
    assert main(["--backend", "mock", "--topic", TOPIC, "--rounds", "1", "--no-abstract",
                 "--log-dir", str(tmp_path)]) == 0
    md = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "## 초록" not in md
