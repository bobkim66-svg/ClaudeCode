"""프로토콜 순수 로직 테스트 (외부 연결 없음)."""
import pytest

from council.config import CouncilConfig
from council.protocol import DiscussionState, Utterance, build_turn_prompt, is_termination
from council.roles import DEFAULT_ROLES, Role, build_seed_prompt, validate_roles

CFG = CouncilConfig(max_rounds=5)
P, C, A, M = DEFAULT_ROLES  # 제안자, 비평가, 대안 설계자, 사회자
TOPIC = "드론 군집의 충돌 회피 방식"


def _u(r, role, text):
    return Utterance(r, role.key, role.name, text)


# ---------- 델타 버퍼 ----------

def test_delta_first_speaker_is_empty():
    assert DiscussionState().delta_for(P.key) == []


def test_delta_excludes_self_and_already_seen():
    s = DiscussionState()
    for role in DEFAULT_ROLES:
        s.record(_u(1, role, f"{role.key}-r1"))
    # 라운드 2: 제안자는 자기 발언 이후의 C, A, M 만 받아야 함
    assert [u.text for u in s.delta_for(P.key)] == ["critic-r1", "pragmatist-r1", "moderator-r1"]
    s.record(_u(2, P, "proposer-r2"))
    # 비평가: 자기 r1 발언 이후 A-r1, M-r1, P-r2
    assert [u.text for u in s.delta_for(C.key)] == ["pragmatist-r1", "moderator-r1", "proposer-r2"]
    assert s.delta_for(P.key) == []


def test_every_utterance_delivered_exactly_once_to_each_other_role():
    """불변 조건: stateful 모드에서 각 역할은 다른 사람의 발언을 정확히 1번씩 받는다 (누락·중복 없음)."""
    s = DiscussionState()
    received = {r.key: [] for r in DEFAULT_ROLES}
    n_rounds = 4
    for rnd in range(1, n_rounds + 1):
        for role in DEFAULT_ROLES:
            received[role.key] += [id(u) for u in s.delta_for(role.key)]
            s.record(_u(rnd, role, f"{role.key}-{rnd}"))
    for role in DEFAULT_ROLES:
        others = [id(u) for u in s.utterances if u.speaker_key != role.key]
        unseen_tail = [id(u) for u in s.delta_for(role.key)]  # 마지막 발언 이후 아직 전달 전
        assert received[role.key] + unseen_tail == others


# ---------- 프롬프트 조립 ----------

def test_stateful_seed_only_on_first_turn():
    s = DiscussionState()
    p1 = build_turn_prompt(P, DEFAULT_ROLES, s, 1, TOPIC, CFG, stateful=True)
    assert "[토론 규칙]" in p1 and TOPIC in p1 and "(없음)" in p1
    s.record(_u(1, P, "안 A"))
    for role in (C, A, M):
        s.record(_u(1, role, f"{role.name} 의견"))
    p2 = build_turn_prompt(P, DEFAULT_ROLES, s, 2, TOPIC, CFG, stateful=True)
    assert "[토론 규칙]" not in p2
    assert "비평가 의견" in p2 and "안 A" not in p2


def _simulate_prompt_lengths(stateful, n_rounds=8, reply_len=600):
    s = DiscussionState()
    lens = []
    for rnd in range(1, n_rounds + 1):
        for role in DEFAULT_ROLES:
            lens.append(len(build_turn_prompt(role, DEFAULT_ROLES, s, rnd, TOPIC,
                                              CouncilConfig(max_rounds=n_rounds), stateful)))
            s.record(_u(rnd, role, "가" * reply_len))
    return lens


@pytest.mark.parametrize("stateful", [True, False])
def test_prompt_length_bounded(stateful):
    """입력 길이가 라운드에 따라 커지지 않아야 함 (O(N^2) 누적 방지).

    상한 근거: stateful 은 델타 = 다른 3명 발언, stateless 는 seed + 요약 1 + 요약 이후 최대 3개.
    따라서 라운드 2 이후 매 턴 입력 ≤ seed + 4 * (발언 + 머리글) 정도로 일정해야 한다.
    """
    lens = _simulate_prompt_lengths(stateful)
    per_round = [lens[i:i + 4] for i in range(0, len(lens), 4)]
    for mid in per_round[2:-1]:
        assert mid == per_round[1]  # 라운드 2 이후 매 라운드 입력 길이가 정확히 일정
    # 마지막 라운드는 사회자에게 '최종 결론' 안내 문구만 추가됨
    final_note = len(" 마지막 라운드이므로 최종 결론을 정리하세요.")
    assert per_round[-1][:-1] == per_round[1][:-1]
    assert per_round[-1][-1] == per_round[1][-1] + final_note
    assert max(lens) < 4 * 700 + 1500


def test_stateless_contains_summary_and_drops_older():
    s = DiscussionState()
    for role in DEFAULT_ROLES:
        s.record(_u(1, role, f"{role.key}-r1"))
    s.record(_u(2, P, "proposer-r2"))
    p = build_turn_prompt(C, DEFAULT_ROLES, s, 2, TOPIC, CFG, stateful=False)
    assert "[토론 규칙]" in p  # stateless 는 매번 seed 포함
    assert "moderator-r1" in p and "proposer-r2" in p
    assert "proposer-r1" not in p and "pragmatist-r1" not in p  # 요약 이전 발언은 요약으로 대체


def test_stateless_before_any_summary_sends_all():
    s = DiscussionState()
    s.record(_u(1, P, "proposer-r1"))
    p = build_turn_prompt(C, DEFAULT_ROLES, s, 1, TOPIC, CFG, stateful=False)
    assert "(아직 없음)" in p and "proposer-r1" in p


def test_last_round_moderator_gets_final_instruction():
    s = DiscussionState()
    p = build_turn_prompt(M, DEFAULT_ROLES, s, CFG.max_rounds, TOPIC, CFG, stateful=True)
    assert "최종 결론" in p


# ---------- 종료 토큰 ----------

def test_end_token_only_in_moderator_seed():
    for role in DEFAULT_ROLES:
        seed = build_seed_prompt(role, DEFAULT_ROLES, TOPIC, CFG)
        assert (CFG.end_token in seed) == role.is_moderator


@pytest.mark.parametrize("reply,expected", [
    ("요약...\n[DISCUSSION_COMPLETE]", True),
    ("요약...\n[DISCUSSION_COMPLETE]\n\n  ", True),           # 뒤쪽 빈 줄
    ("요약...\n**[DISCUSSION_COMPLETE]**", True),             # 마크다운 강조
    ("요약...\n`[DISCUSSION_COMPLETE]`.", True),
    ("[DISCUSSION_COMPLETE] 는 아직 이르다\n남은 쟁점 2개", False),  # 본문 중간 언급
    ("결론: [DISCUSSION_COMPLETE] 하겠습니다", False),          # 같은 줄에 다른 글
    ("", False),
    ("\n\n", False),
])
def test_termination_detection(reply, expected):
    assert is_termination(M, reply, CFG) is expected


def test_termination_ignored_for_non_moderator():
    assert is_termination(C, "반박\n[DISCUSSION_COMPLETE]", CFG) is False


# ---------- 설정/역할 검증 ----------

def test_config_validation():
    with pytest.raises(ValueError):
        CouncilConfig(max_rounds=0)
    with pytest.raises(ValueError):
        CouncilConfig(end_token="  ")


def test_validate_roles():
    validate_roles(DEFAULT_ROLES)
    with pytest.raises(ValueError):
        validate_roles(())
    with pytest.raises(ValueError):
        validate_roles((M, P))  # 사회자가 마지막이 아님
    with pytest.raises(ValueError):
        validate_roles((P, P, M))  # key 중복
    with pytest.raises(ValueError):
        validate_roles((P, C))  # 사회자 없음
    validate_roles((Role("a", "가", "x"), Role("m", "사회", "y", is_moderator=True)))  # 2인 구성 허용
