"""형상 처리 (Qt 없음 → pytest 로 검사).

1) 격자화: 물체·송신기 → 셀 중심 기준 bool 마스크 (계단 근사). 물체의 경계상자 안쪽 부분 격자만 계산해 메모리 절약.
2) STL: 바이너리/ASCII 읽기, 회전·배율·이동, 광선 홀짝(parity) 판정으로 내부 셀 채우기 (닫힌 메시 필요).
3) 표시용 삼각형 메시 (verts (V,3), faces (F,3)): 직육면체, 원기둥, 구, 원판, STL.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..config import ConfigError
from .config3d import Body, Transducer

CYL_SEGMENTS = 40
SPHERE_RINGS, SPHERE_SEGMENTS = 18, 32
# STL 광선이 삼각형 모서리·꼭짓점을 정확히 지나면 교차를 두 번 세거나 놓친다. 광선을 셀 크기의 무리수배만큼
# 살짝 옮겨 그 경우를 피한다 (마스크 오차는 셀 크기의 1e-6 배 수준).
_RAY_JITTER = (1.234567e-6 * math.pi, 2.345678e-6 * math.e)


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------
def load_stl(path: str | Path) -> np.ndarray:
    """STL → 삼각형 (n,3,3) [파일 단위]. 바이너리/ASCII 자동 판별."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"STL 파일을 찾을 수 없습니다: {p}")
    data = p.read_bytes()
    if len(data) >= 84:
        n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
        if len(data) == 84 + 50 * n and n > 0:
            rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(float)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ConfigError(f"STL 형식을 읽을 수 없습니다 (바이너리 크기 불일치, ASCII 아님): {p.name}") from None
    verts = [[float(v) for v in line.split()[1:4]] for line in text.splitlines() if line.strip().startswith("vertex")]
    if not verts or len(verts) % 3:
        raise ConfigError(f"ASCII STL 에서 삼각형을 찾지 못했습니다: {p.name}")
    return np.asarray(verts).reshape(-1, 3, 3)


def write_stl_ascii(path: str | Path, tris: np.ndarray, name: str = "polarsim") -> None:
    """삼각형 (n,3,3) → ASCII STL (테스트·예시용)."""
    lines = [f"solid {name}"]
    for t in tris:
        nrm = np.cross(t[1] - t[0], t[2] - t[0])
        nn = np.linalg.norm(nrm)
        nrm = nrm / nn if nn > 0 else nrm
        lines += [f"facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}", "outer loop"]
        lines += [f"vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}" for v in t]
        lines += ["endloop", "endfacet"]
    lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


_STL_CACHE: dict[tuple[str, float], np.ndarray] = {}


def body_triangles(body: Body) -> np.ndarray:
    """물체 → 전역 좌표 삼각형 (n,3,3) [m] (표시·STL 격자화 공용)."""
    if body.kind == "stl":
        p = Path(body.path)
        key = (str(p.resolve()), p.stat().st_mtime if p.exists() else 0.0)
        if key not in _STL_CACHE:
            _STL_CACHE[key] = load_stl(p)
        local = _STL_CACHE[key] * body.size[0]
    else:
        v, f = _primitive_mesh_local(body)
        local = v[f]
    return local @ body.R.T + np.asarray(body.center)


def _primitive_mesh_local(body: Body) -> tuple[np.ndarray, np.ndarray]:
    if body.kind == "box":
        return box_mesh(body.size)
    if body.kind == "cylinder":
        return cylinder_mesh(body.size[0], body.size[1] / 2)
    return sphere_mesh(body.size[0] / 2)


# ---------------------------------------------------------------------------
# 표시용 메시 (국소 좌표, 중심 원점)
# ---------------------------------------------------------------------------
def box_mesh(size) -> tuple[np.ndarray, np.ndarray]:
    lx, ly, lz = (s / 2 for s in size)
    v = np.array([[sx * lx, sy * ly, sz * lz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    # 꼭짓점 번호 = 4·(x>0) + 2·(y>0) + (z>0). 바깥을 향하는 반시계 순서.
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def cylinder_mesh(length: float, radius: float, n: int = CYL_SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
    """축 = 국소 x."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.zeros(n), radius * np.cos(a), radius * np.sin(a)], -1)
    h = np.array([length / 2, 0, 0])
    v = np.vstack([ring - h, ring + h, [-h], [h]])
    c0, c1 = 2 * n, 2 * n + 1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i], [c0, j, i], [c1, n + i, n + j]]
    return v, np.array(f)


def sphere_mesh(radius: float, rings: int = SPHERE_RINGS, segs: int = SPHERE_SEGMENTS):
    th = np.linspace(0, np.pi, rings + 1)
    ph = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    v = radius * np.stack([np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T)], -1).reshape(-1, 3)
    f = []
    for r in range(rings):
        for s in range(segs):
            a, b = r * segs + s, r * segs + (s + 1) % segs
            c, d = a + segs, b + segs
            f += [[a, c, d], [a, d, b]]
    return v, np.array(f)


def transducer_mesh(t: Transducer) -> tuple[np.ndarray, np.ndarray]:
    """송신기 원판 (전역 좌표)."""
    v, f = cylinder_mesh(t.thickness, t.diameter / 2)
    from .config3d import rotation

    return v @ rotation(t.az_deg, t.el_deg).T + np.asarray(t.center), f


def body_mesh(body: Body) -> tuple[np.ndarray, np.ndarray]:
    """물체 (전역 좌표). STL 은 삼각형을 그대로 펼친다."""
    if body.kind == "stl":
        tris = body_triangles(body)
        return tris.reshape(-1, 3), np.arange(tris.shape[0] * 3).reshape(-1, 3)
    v, f = _primitive_mesh_local(body)
    return v @ body.R.T + np.asarray(body.center), f


# ---------------------------------------------------------------------------
# 격자화
# ---------------------------------------------------------------------------
def _block(xc, yc, zc, lo, hi):
    """경계상자 [lo, hi] 안의 셀 중심 인덱스 범위 (슬라이스 3개). 비어 있으면 None."""
    out = []
    for c, a, b in zip((xc, yc, zc), lo, hi):
        i0, i1 = np.searchsorted(c, a, "left"), np.searchsorted(c, b, "right")
        if i1 <= i0:
            return None
        out.append(slice(int(i0), int(i1)))
    return tuple(out)


def _local_coords(xc, yc, zc, sl, center, R):
    X = xc[sl[0]][:, None, None] - center[0]
    Y = yc[sl[1]][None, :, None] - center[1]
    Z = zc[sl[2]][None, None, :] - center[2]
    # 국소 = Rᵀ (전역 − 중심)
    return (R[0, 0] * X + R[1, 0] * Y + R[2, 0] * Z,
            R[0, 1] * X + R[1, 1] * Y + R[2, 1] * Z,
            R[0, 2] * X + R[1, 2] * Y + R[2, 2] * Z)


def body_mask(body: Body, xc: np.ndarray, yc: np.ndarray, zc: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    """out (nx,ny,nz) bool 에 물체 셀을 OR 로 더한다. 셀 중심이 물체 안이면 강체.
    반환 (덮은 셀 수, STL 홀수 교차 기둥 수 — 0 이 아니면 메시가 닫혀 있지 않음)."""
    tris = body_triangles(body)
    pts = tris.reshape(-1, 3)
    sl = _block(xc, yc, zc, pts.min(0), pts.max(0))
    if sl is None:
        return 0, 0
    odd = 0
    if body.kind == "stl":
        m, odd = voxelize(tris, xc[sl[0]], yc[sl[1]], zc[sl[2]])
    else:
        lx, ly, lz = _local_coords(xc, yc, zc, sl, np.asarray(body.center), body.R)
        if body.kind == "box":
            a, b, c = (s / 2 for s in body.size)
            m = (np.abs(lx) <= a) & (np.abs(ly) <= b) & (np.abs(lz) <= c)
        elif body.kind == "cylinder":
            m = (np.abs(lx) <= body.size[0] / 2) & (ly**2 + lz**2 <= (body.size[1] / 2) ** 2)
        else:
            m = lx**2 + ly**2 + lz**2 <= (body.size[0] / 2) ** 2
    out[sl] |= m
    return int(m.sum()), odd


def transducer_cells(t: Transducer, xc, yc, zc):
    """송신기 원판 셀. 반환 (슬라이스 3개, 블록 마스크) 또는 None."""
    c = np.asarray(t.center)
    r = t.diameter / 2 + t.thickness
    sl = _block(xc, yc, zc, c - r, c + r)
    if sl is None:
        return None
    from .config3d import rotation

    d, ly, lz = _local_coords(xc, yc, zc, sl, c, rotation(t.az_deg, t.el_deg))  # d = 법선 방향 거리
    m = (np.abs(d) <= t.thickness / 2) & (ly**2 + lz**2 <= (t.diameter / 2) ** 2)
    return sl, m


def voxelize(tris: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, int]:
    """닫힌 삼각형 메시 → 셀 중심 (xs, ys, zs) 이 내부인 bool (nx,ny,nz) 와 홀수 교차 기둥 수 (0 이면 닫힌 메시).

    각 (x, y) 기둥에서 −z 쪽 무한원에서 +z 로 쏜 광선이 셀 중심 아래에서 면을 홀수 번 지나면 내부 (조르당 곡면 정리).
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    if min(nx, ny, nz) == 0 or len(tris) == 0:
        return np.zeros((nx, ny, nz), bool), 0
    dx = float(np.median(np.diff(xs))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if ny > 1 else 1.0
    qx, qy = xs + _RAY_JITTER[0] * dx, ys + _RAY_JITTER[1] * dy
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    i0 = np.searchsorted(qx, np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0]), "left")
    i1 = np.searchsorted(qx, np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0]), "right")
    j0 = np.searchsorted(qy, np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1]), "left")
    j1 = np.searchsorted(qy, np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1]), "right")
    cols, zhit = [], []
    for t in np.nonzero((i1 > i0) & (j1 > j0))[0]:
        X, Y = np.meshgrid(qx[i0[t]:i1[t]], qy[j0[t]:j1[t]], indexing="ij")
        (ax_, ay_, az_), (bx_, by_, bz_), (cx_, cy_, cz_) = a[t], b[t], c[t]
        den = (by_ - cy_) * (ax_ - cx_) + (cx_ - bx_) * (ay_ - cy_)
        if den == 0:
            continue  # 광선과 나란한 (수직) 삼각형
        l1 = ((by_ - cy_) * (X - cx_) + (cx_ - bx_) * (Y - cy_)) / den
        l2 = ((cy_ - ay_) * (X - cx_) + (ax_ - cx_) * (Y - cy_)) / den
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        ii, jj = np.nonzero(inside)
        cols.append((ii + i0[t]) * ny + (jj + j0[t]))
        zhit.append(l1[inside] * az_ + l2[inside] * bz_ + l3[inside] * cz_)
    if not cols:
        return np.zeros((nx, ny, nz), bool), 0
    col = np.concatenate(cols)
    z = np.concatenate(zhit)
    hist = np.zeros((nx * ny, nz + 1), np.int32)
    np.add.at(hist, (col, np.searchsorted(zs, z, "left")), 1)  # 칸 k: zs[k−1] < z ≤ zs[k]
    below = np.cumsum(hist, axis=1)[:, :nz]  # 셀 k 중심 아래 교차 수
    odd_cols = int((np.bincount(col, minlength=nx * ny) % 2).sum())
    return (below % 2 == 1).reshape(nx, ny, nz), odd_cols
