"""역할(페르소나) 정의와 첫 프롬프트(seed) 생성."""
from __future__ import annotations

from dataclasses import dataclass

from .config import CouncilConfig


@dataclass(frozen=True)
class Role:
    """토론 참가자 1명.

    Attributes:
        key: 내부 식별자 (영문, 고유).
        name: 회의록/프롬프트에 쓰는 표시 이름.
        mission: 역할 설명 (seed 프롬프트에 들어감).
        model_hint: 백엔드가 고를 모델 종류 힌트. "fast" | "thinking".
        is_moderator: True 면 라운드 요약과 종료 판단을 맡는다. 종료 토큰은 이 역할에게만 알려준다.
        extra_rules: 이 역할에만 추가되는 규칙 (예: 비평가의 '치명적 약점' 의무).
    """

    key: str
    name: str
    mission: str
    model_hint: str = "fast"
    is_moderator: bool = False
    extra_rules: tuple[str, ...] = ()


DEFAULT_ROLES: tuple[Role, ...] = (
    Role(
        key="proposer",
        name="제안자",
        mission="주제에 대한 초기 설계안과 구체적 구현 방안을 제시하고, 이후 라운드에서는 지적을 반영해 안을 갱신한다.",
    ),
    Role(
        key="critic",
        name="비평가",
        mission="제안된 안의 논리적 허점, 예외 상황, 물리적·비용적 한계를 근거와 함께 엄격하게 검증한다.",
        model_hint="thinking",
        # 첫 실사용에서 비평이 '부분 동의 + 의향 질문'으로 매 라운드 무뎌짐 → 강제 장치
        extra_rules=(
            "매 발언에 '치명적 약점:' 한 줄로 현재 안의 가장 치명적인 약점 1개를 반드시 명시한다. "
            "이전 약점이 해결됐다면 해결됐다고 쓰고 새 약점을 찾는다.",
            "근거 없이 동의하지 않는다. 안 전체의 전제(주제 선택 자체 포함)도 공격 대상이다.",
            "'~할 의향이 있습니까?' 같은 질문형으로 끝내지 말고, 통과 조건을 명확한 요구사항으로 제시한다.",
        ),
    ),
    Role(
        key="pragmatist",
        name="대안 설계자",
        mission="비평가의 지적을 반영해 현실적으로 실행 가능한 수정안과 대안을 도출한다.",
    ),
    Role(
        key="moderator",
        name="사회자",
        mission="이번 라운드의 핵심 쟁점을 5줄 이내로 요약하고, 남은 쟁점과 토론 수렴 여부를 판단한다.",
        is_moderator=True,
    ),
)


# 토론 유형별 점검 항목 (CouncilConfig.agenda 로 전달)
AGENDAS: dict[str, tuple[str, ...]] = {
    "general": (),
    "topic-review": (
        "새로운 점: 선행연구 대비 무엇이 새로운가 (확인되지 않은 문헌은 '확인 필요'로 표시)",
        "실현 가능성: 학위 기간, 장비·플랫폼, 연산 자원, 데이터로 완수 가능한가",
        "위험: 기술·안전·일정 측면에서 실패할 수 있는 지점과 대비책",
        "대안 비교: 범위를 줄이거나 바꾼 대안 주제 1개 이상과 비교했을 때 이 주제가 나은가",
    ),
}


def validate_roles(roles: tuple[Role, ...]) -> None:
    """역할 목록의 불변 조건 검사: 비어 있지 않음, key 고유, 사회자 정확히 1명이며 마지막 순서."""
    if not roles:
        raise ValueError("역할이 하나도 없음")
    keys = [r.key for r in roles]
    if len(set(keys)) != len(keys):
        raise ValueError(f"역할 key 중복: {keys}")
    mods = [r for r in roles if r.is_moderator]
    if len(mods) != 1:
        raise ValueError(f"사회자는 정확히 1명이어야 함 (현재 {len(mods)}명)")
    if not roles[-1].is_moderator:
        raise ValueError("사회자는 라운드의 마지막 발언자여야 함 (요약·종료 판단을 위해)")


def build_seed_prompt(role: Role, roles: tuple[Role, ...], topic: str, cfg: CouncilConfig) -> str:
    """역할에게 처음 한 번 주는 규칙 + 페르소나 프롬프트.

    - 종료 토큰은 사회자 seed 에만 넣는다 (다른 참가자가 인용해 토론이 잘못 끝나는 것 방지).
    - 전체 라운드 수는 알려주지 않는다. 예산을 알면 마지막 라운드에 맞춰 억지로 수렴하는 경향이
      첫 실사용 회의록에서 관찰됨 (1/3 '수렴 전' → 3/3 정확히 종료).
    - 수치는 출처·계산이 있을 때만 쓰게 한다 (수치 의무 규칙이 출처 없는 수치를 만들어 냄).
    """
    members = ", ".join(r.name for r in roles)
    lines = [
        "당신은 여러 AI가 참여하는 구조화된 토론의 참가자입니다.",
        f"[토론 주제] {topic}",
        f"[참가자 발언 순서] {members}",
        f"[당신의 역할] {role.name}: {role.mission}",
    ]
    if cfg.agenda:
        lines += ["", "[이번 토론의 점검 항목] 모든 참가자는 아래 항목을 다뤄야 한다."]
        lines += [f"  {i}. {item}" for i, item in enumerate(cfg.agenda, 1)]
    lines += [
        "",
        "[토론 규칙]",
        "1. 인사말, 칭찬, 감사, 감정적 동조 표현과 '전폭 수용' 같은 수사를 쓰지 않는다. "
        "받아들이는 점은 무엇을 왜 받아들이는지 한 줄로만 쓴다.",
        "2. 수치는 출처나 계산 과정이 있을 때만 쓰고, 추정치는 '(추정)'이라고 표시한다. "
        "출처 없는 수치를 지어내지 않는다. 정성적 근거(논리, 메커니즘, 반례)도 충분한 근거다.",
        f"3. 공백 포함 {cfg.char_limit}자 이내로 쓴다.",
    ]
    if role.is_moderator:
        fmt = "(1) 라운드 요약: 핵심 쟁점 5줄 이내 (2) 남은 쟁점"
        if cfg.agenda:
            fmt += " (점검 항목별로 '해결/부분/미해결'과 이유 한 줄)"
        lines += [
            f"4. 형식: {fmt} (3) 다음 라운드에 다룰 질문",
            "5. 토론에는 라운드 수 제한이 없다고 가정하고, 실제로 남은 쟁점만 보고 판단한다. "
            "참가자들이 서로 동의하는 것만으로 수렴으로 보지 않는다.",
            "6. 미해결 쟁점이 하나도 없다고 판단될 때만"
            + (" (점검 항목이 모두 '해결'일 때만)" if cfg.agenda else "")
            + f", 최종 결론을 정리한 뒤 답변의 맨 마지막 줄에 {cfg.end_token} 만 단독으로 출력한다. "
            "그 외의 경우에는 이 문자열을 절대 쓰지 않는다.",
        ]
    else:
        lines += [
            "4. 모든 발언은 3단 구조로만 쓴다: (1) 입장: 동의/부분 동의/반박 중 하나 "
            "(2) 근거 (3) 제안: 수정안 또는 요구사항. 불릿은 5개 이내.",
            "5. 다른 참가자의 발언을 그대로 되풀이하지 않는다.",
        ]
    n = len([ln for ln in lines if ln[:1].isdigit()])
    for k, rule in enumerate(role.extra_rules, n + 1):
        lines.append(f"{k}. {rule}")
    return "\n".join(lines)
