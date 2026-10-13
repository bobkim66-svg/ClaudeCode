"""결과 저장 (CSV, JSON). GUI 와 독립이라 pytest 로 검사한다.

- 결과 폴더는 타임스탬프를 붙여 새로 만든다 (덮어쓰기 방지).
- 텍스트 파일은 UTF-8 로 쓴다. CSV 는 Windows Excel 이 한글 헤더를 바로 읽도록 BOM 이 붙은 'utf-8-sig'.
- NaN (정의 불가 지표) 은 CSV 에 'nan' 으로 그대로 남긴다 (임의 값으로 채우지 않음).
"""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Medium, MetricSettings, ObservationPlane, SimConfig, Source, TimeWindow
from .metrics import SimulationResult
from .physics import plane_basis


def timestamped_dir(base: Path, prefix: str = "polarsim") -> Path:
    """base/prefix_YYYYmmdd_HHMMSS[_n] 를 새로 만들어 반환. 같은 초에 두 번 저장해도 덮어쓰지 않는다."""
    base = Path(base)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / f"{prefix}_{stamp}"
    n = 1
    while path.exists():
        path = base / f"{prefix}_{stamp}_{n}"
        n += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def config_to_dict(config: SimConfig) -> dict:
    return dataclasses.asdict(config)


def config_from_dict(d: dict) -> SimConfig:
    """config_to_dict 의 역변환 (JSON 에서 리스트로 바뀐 벡터를 튜플로 되돌림)."""

    def tup(v):
        return None if v is None else tuple(float(x) for x in v)

    sources = tuple(
        Source(**{**s, "direction": tup(s["direction"]), "ref_point": tup(s["ref_point"])}) for s in d["sources"]
    )
    p = d["plane"]
    plane = ObservationPlane(**{**p, "origin": tup(p["origin"]), "normal": tup(p["normal"]), "u_axis": tup(p["u_axis"])})
    return SimConfig(
        medium=Medium(**d["medium"]),
        sources=sources,
        plane=plane,
        window=TimeWindow(**d["window"]),
        metrics=MetricSettings(**d["metrics"]),
    )


def write_config_json(config: SimConfig, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(config_to_dict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


CSV_COLUMNS = [
    "i", "j", "s_u [m]", "s_v [m]", "x [m]", "y [m]", "z [m]",
    "L", "e", "planarity", "T", "Ix [W/m2]", "Iy [W/m2]", "Iz [W/m2]", "rms [m]",
]


def write_metrics_csv(result: SimulationResult, plane: ObservationPlane, path: Path) -> Path:
    """입자별 지표를 한 줄씩. result 는 평면 격자 결과 (shape (nx, ny))."""
    path = Path(path)
    e_u, e_v, _ = plane_basis(plane)
    rel = result.points - np.asarray(plane.origin, float)
    su, sv = rel @ e_u, rel @ e_v
    nx, ny = result.L.shape
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for i in range(nx):
            for j in range(ny):
                row = [i, j, su[i, j], sv[i, j], *result.points[i, j],
                       result.L[i, j], result.e[i, j], result.planarity[i, j], result.T[i, j],
                       *result.intensity[i, j], result.rms[i, j]]
                w.writerow([f"{v:.10g}" if isinstance(v, float | np.floating) else v for v in row])
    return path


def write_summary(result: SimulationResult, path: Path) -> Path:
    """검증 수치와 경고를 사람이 읽는 텍스트로."""
    lines = [
        f"입자 수 n = {result.L.size}",
        f"시간창 = {result.window.duration:.6g} s, 표본 {result.window.n_samples} 개, "
        f"공통 주기 = {result.window.common_period}",
        f"회전 검증 max|curl u|/max|grad u| = {result.curl_ratio:.3e} (엔진 A: 구현 버그 검출용)",
        f"A_ref = {result.a_ref:.4g} m, I_ref = {result.i_ref:.4g} W/m2",
    ]
    for k in ("L", "e", "planarity", "T"):
        a = getattr(result, k)
        ok = a[~np.isnan(a)]
        rng = f"{ok.min():.4f} ~ {ok.max():.4f}" if ok.size else "전부 NaN"
        lines.append(f"{k}: 범위 {rng}, NaN {int(np.isnan(a).sum())}/{a.size}")
    lines += ["경고:"] + [f"  - {w}" for w in result.warnings] if result.warnings else ["경고: 없음"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)
