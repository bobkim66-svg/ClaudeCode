"""엔진 3D: 해석해 비교, 에너지 보존, 반사, 대칭, 재현성, 형상 처리 (STL 포함)."""

import math

import numpy as np
import pytest

from polarsim.config import ConfigError
from polarsim.engine3d.analytic3d import duct_plane_wave, piston_on_axis
from polarsim.engine3d.config3d import Body, Signal, Sim3DConfig, Transducer, direction, rotation
from polarsim.engine3d.fdtd3d import Cancelled, estimate, run, setup_grid
from polarsim.engine3d.geometry3d import (body_mask, body_mesh, box_mesh, cylinder_mesh, load_stl, sphere_mesh,
                                          voxelize, write_stl_ascii)

F = 5000.0
W = 2 * math.pi * F
RHO, C = 1000.0, 1500.0
XI = 1e-6


def duct(kind="cw", rigid_end=False, amplitude=XI, **kw):
    """단면 0.3 m × 0.3 m 강체 관, 바닥 전체가 피스톤 (원판 지름이 단면 대각선보다 큼)."""
    return Sim3DConfig(signal=Signal(kind, F, cycles=2), domain=(-0.15, 0.15, -0.15, 0.15, -0.06, 1.2),
                       pml_sides=(False, False, False, False, False, not rigid_end),
                       transducers=(Transducer(center=(0, 0, -0.03), el_deg=90, diameter=1.0, thickness=0.06,
                                               amplitude=amplitude),), **kw)


# ---- 해석해 ----------------------------------------------------------------------
def test_duct_plane_wave_amplitude_and_linearity():
    r = run(duct())
    num = np.abs(r.P[2, 2, r.zc > 0])
    assert num == pytest.approx(duct_plane_wave(RHO, C, W * XI), rel=2e-3)  # 진폭 + PML 무반사 (맥놀이 없음)
    assert r.convergence < 1e-3
    r2 = run(duct(amplitude=2 * XI))
    assert np.allclose(r2.P, 2 * r.P, rtol=1e-4, atol=1e-6 * np.abs(r.P).max())  # 선형


def test_duct_phase_speed():
    """관 속 위상: arg P 가 z 에 대해 기울기 k_num 으로 변해야 한다. 이산 분산 (π/N)²/6 이내."""
    r = run(duct())
    z = r.zc[(r.zc > 0.05) & (r.zc < 0.9)]
    ph = np.unwrap(np.angle(r.P[2, 2, (r.zc > 0.05) & (r.zc < 0.9)]))
    k_num = np.polyfit(z, ph, 1)[0]
    k = W / C
    assert k_num == pytest.approx(k, rel=(math.pi / 10) ** 2 / 6 * 1.5)


def test_baffled_piston_on_axis_matches_rayleigh():
    """무한 배플 원형 피스톤 축 위 |p| (닫힌 해) 와 비교. 오차는 격자 세분화로 줄어든다 (10→16→24 셀/λ: 5.9→2.2→0.4%)."""
    a = 0.225
    cfg = Sim3DConfig(signal=Signal("cw", F), domain=(-0.45, 0.45, -0.45, 0.45, -0.06, 0.9), cells_per_wavelength=16,
                      pml_sides=(True, True, True, True, False, True),
                      bodies=(Body("box", (0, 0, -0.03), (4, 4, 0.06), name="배플"),),
                      transducers=(Transducer(center=(0, 0, -0.03), el_deg=90, diameter=2 * a, thickness=0.06),))
    r = run(cfg)
    i, j = np.argmin(np.abs(r.xc)), np.argmin(np.abs(r.yc))
    sel = r.zc > 0.3
    num = np.abs(r.P[i, j, sel])
    ana = piston_on_axis(r.zc[sel], a, W / C, RHO, C, W * XI)
    assert np.sqrt(np.mean((num - ana) ** 2)) / ana.mean() < 0.035
    assert r.curl_ratio < 1e-4  # 비회전 (float32 반올림 수준)


# ---- 핑: 에너지, 반사, 방향 -------------------------------------------------------
@pytest.fixture(scope="module")
def ping_tilted():
    cfg = Sim3DConfig(signal=Signal("ping", F, cycles=3), domain=(-0.8, 0.8, -0.8, 0.8, -0.8, 0.8),
                      cells_per_wavelength=10,
                      transducers=(Transducer(center=(-0.4, 0, 0), az_deg=20, el_deg=10, diameter=0.3,
                                              thickness=0.08),))
    return cfg, run(cfg)


def test_ping_energy_balance(ping_tilted):
    """송신기가 한 일 W = 영역 안 음향 에너지 E (펄스가 다 나온 뒤, PML 에 닿기 전). 그 뒤 PML 이 흡수해 E → 0."""
    cfg, r = ping_tilted
    e = r.energy
    t_emit = cfg.signal.duration
    k = int(np.argmin(np.abs(e["t"] - (t_emit + 0.05e-3))))
    assert e["E"][k] / e["W"][k] == pytest.approx(1.0, abs=0.03)
    assert e["E"][-1] < 1e-4 * e["W"][-1]
    assert r.curl_ratio < 1e-4


def test_ping_beam_follows_transducer_normal(ping_tilted):
    cfg, r = ping_tilted
    t = cfg.transducers[0]
    X, Y, Z = np.meshgrid(r.xc, r.yc, r.zc, indexing="ij")
    d = np.stack([X - t.center[0], Y - t.center[1], Z - t.center[2]], -1)
    rr = np.linalg.norm(d, axis=-1)
    shell = (rr > 0.7) & (rr < 0.9)
    k = np.argmax(np.where(shell, np.nan_to_num(r.p_max), 0))
    v = d.reshape(-1, 3)[k] / rr.ravel()[k]
    ang = math.degrees(math.acos(np.clip(v @ t.normal, -1, 1)))
    assert ang < 8.0  # 근거리 + 계단 근사 + 셀 표본 (빔 폭 ~ 30°)


def test_rigid_wall_reflects_ping_fully():
    """관 끝 강체 벽: 반사 펄스 에너지 = 입사 펄스 에너지 (|R| = 1), 벽 앞 압력 2 배 (R = +1).

    최대값이 아니라 ∫p² dt 로 비교: 2주기 펄스는 대역이 넓어 수치 분산으로 모양이 조금 바뀌지만 (최대값 ±4%),
    도약법은 무손실이라 에너지는 보존된다."""
    r = run(duct("ping", rigid_end=True, ping_transits=1.5, frames_per_period=20))
    k = int(np.argmin(np.abs(r.lat_pos[:, 2] - 0.6) + np.abs(r.lat_pos[:, 0]) + np.abs(r.lat_pos[:, 1])))
    sig = r.p_frames[:, k].astype(float)
    t_split = (0.6 + 1.2 - 0.6) / C  # 벽에서 돌아온 펄스 앞머리가 탐침에 오기 전 (경로 0.6 + 2·0.6 m 의 중간쯤)
    inc, ref = sig[r.times < t_split], sig[r.times >= t_split]
    assert np.sum(ref**2) / np.sum(inc**2) == pytest.approx(1.0, abs=0.02)
    # 부호: 강체 벽 (R = +1) 바로 앞에서는 입사·반사가 겹쳐 압력이 약 2 배 (R = −1 이면 약 0)
    wall = r.slice_frames[0][:, r.slice_frames[0].shape[1] // 2, -1]  # x 중앙 단면, 벽에 붙은 셀 (벽에서 dx/2)
    assert np.abs(wall).max() / np.abs(inc).max() == pytest.approx(2.0, abs=0.12)


def test_symmetry_and_reproducibility():
    cfg = Sim3DConfig(signal=Signal("cw", F), domain=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), cells_per_wavelength=8,
                      transducers=(Transducer(center=(-0.2, 0, 0), diameter=0.3, thickness=0.1),),
                      bodies=(Body("sphere", (0.2, 0, 0), (0.3, 0, 0)),), transits=1.0)
    r1, r2 = run(cfg), run(cfg)
    assert np.array_equal(r1.P, r2.P)  # 결정적 (난수 없음)
    P = np.abs(r1.P)
    assert np.allclose(P, P[:, ::-1, :], rtol=1e-3, atol=1e-4 * P.max())  # y → −y 대칭
    assert np.allclose(P, P[:, :, ::-1], rtol=1e-3, atol=1e-4 * P.max())


def test_cancel():
    with pytest.raises(Cancelled):
        run(duct(), should_stop=lambda: True)


# ---- 설정 검증 --------------------------------------------------------------------
def test_config_errors():
    t = Transducer(center=(0, 0, 0))
    with pytest.raises(ConfigError, match="켜진 송신기"):
        Sim3DConfig(transducers=(Transducer(enabled=False),)).validate()
    with pytest.raises(ConfigError, match="얇아"):
        Sim3DConfig(transducers=(Transducer(thickness=0.01),)).validate()
    with pytest.raises(ConfigError, match="영역 밖"):
        Sim3DConfig(transducers=(Transducer(center=(5, 0, 0)),)).validate()
    with pytest.raises(ConfigError, match="너무 큽니다"):
        Sim3DConfig(transducers=(t,), domain=(-5, 5, -5, 5, -5, 5), cells_per_wavelength=20).validate()
    with pytest.raises(ConfigError, match="CFL"):
        Sim3DConfig(transducers=(t,), cfl=0.6).validate()
    with pytest.raises(ConfigError, match="크기"):
        Sim3DConfig(transducers=(t,), bodies=(Body("box", size=(1, 0, 1)),)).validate()
    with pytest.raises(ConfigError, match="유체에 닿지"):  # 송신기가 물체 속에 묻힘
        setup_grid(Sim3DConfig(transducers=(t,), bodies=(Body("sphere", size=(1.0, 0, 0)),)))
    est = estimate(Sim3DConfig(transducers=(t,)))
    assert est["cells"] == np.prod(est["shape"]) and est["steps"] > 0


# ---- 형상 ------------------------------------------------------------------------
def _signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6)


def test_meshes_closed_and_outward():
    assert _signed_volume(*box_mesh((1, 2, 3))) == pytest.approx(6.0)
    assert _signed_volume(*cylinder_mesh(2.0, 0.5)) == pytest.approx(math.pi * 0.25 * 2, rel=0.01)
    assert _signed_volume(*sphere_mesh(1.0)) == pytest.approx(4 / 3 * math.pi, rel=0.02)


def test_rotation_and_direction():
    R = rotation(30, 20)
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(R[:, 0], direction(30, 20))
    assert np.allclose(direction(90, 0), [0, 1, 0], atol=1e-12) and np.allclose(direction(0, 90), [0, 0, 1], atol=1e-12)


def _grid(n=40, L=1.0):
    c = (np.arange(n) + 0.5) / n * 2 * L - L
    return c, c, c


def test_rotated_box_mask():
    xs, ys, zs = _grid()
    a = np.zeros((40, 40, 40), bool)
    b = np.zeros_like(a)
    body_mask(Body("box", (0.1, 0, 0), (0.8, 0.4, 0.2)), xs, ys, zs, a)
    body_mask(Body("box", (0.1, 0, 0), (0.4, 0.8, 0.2), az_deg=90), xs, ys, zs, b)  # 국소 x 를 y 로 돌림
    assert a.sum() > 0 and np.array_equal(a, b)


def test_stl_voxelize_matches_primitive(tmp_path):
    xs, ys, zs = _grid()
    for body in (Body("box", (0.05, -0.1, 0.02), (0.7, 0.5, 0.3), az_deg=25, el_deg=-15),
                 Body("sphere", (0, 0.1, 0), (0.8, 0, 0))):
        ref = np.zeros((40, 40, 40), bool)
        body_mask(body, xs, ys, zs, ref)
        v, f = body_mesh(body)
        path = tmp_path / "m.stl"
        write_stl_ascii(path, v[f])
        got = np.zeros_like(ref)
        _, odd = body_mask(Body("stl", (0, 0, 0), (1.0, 0, 0), path=str(path)), xs, ys, zs, got)
        assert odd == 0
        # 구는 다면체 근사라 표면 셀 몇 개만 다를 수 있음
        assert np.sum(ref ^ got) <= (0 if body.kind == "box" else 0.03 * ref.sum())


def test_stl_binary_scale_offset_and_open_mesh(tmp_path):
    v, f = box_mesh((100.0, 50.0, 40.0))  # mm 단위 STL
    tris = v[f].astype("<f4")
    rec = np.zeros(len(tris), dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["v"] = tris
    path = tmp_path / "box_mm.stl"
    path.write_bytes(b"\0" * 80 + np.uint32(len(tris)).tobytes() + rec.tobytes())
    assert np.allclose(load_stl(path), tris)
    xs, ys, zs = _grid()
    got = np.zeros((40, 40, 40), bool)
    body_mask(Body("stl", (0.2, 0, 0), (0.001, 0, 0), path=str(path)), xs, ys, zs, got)  # mm → m, x 로 0.2 m 이동
    ref = np.zeros_like(got)
    body_mask(Body("box", (0.2, 0, 0), (0.1, 0.05, 0.04)), xs, ys, zs, ref)
    assert np.array_equal(got, ref)
    m, odd = voxelize(tris[:-1].astype(float) * 0.01, xs, ys, zs)  # 삼각형 하나 뺀 열린 메시
    assert odd > 0


# ---- 설정 파일·예시 ------------------------------------------------------------------
def test_json_roundtrip_with_relative_stl(tmp_path):
    from polarsim.engine3d.config3d import from_dict, presets, to_dict

    stl = tmp_path / "parts" / "hull.stl"
    stl.parent.mkdir()
    v, f = box_mesh((0.4, 0.2, 0.2))
    write_stl_ascii(stl, v[f])
    cfg = presets()["seabed_ping"].config
    cfg = Sim3DConfig(**{**cfg.__dict__, "bodies": cfg.bodies + (Body("stl", (0.3, 0, 0), (1.0, 0, 0), path=str(stl)),)})
    d = to_dict(cfg, base_dir=tmp_path)
    assert d["bodies"][-1]["path"] == "parts/hull.stl"  # 폴더째 옮겨도 열리게 상대경로
    moved = tmp_path / "moved"
    (moved / "parts").mkdir(parents=True)
    (moved / "parts" / "hull.stl").write_bytes(stl.read_bytes())
    back = from_dict(d, base_dir=moved)
    assert back.bodies[-1].path == str((moved / "parts" / "hull.stl").resolve())
    assert back.signal == cfg.signal and back.transducers == cfg.transducers and back.domain == cfg.domain
    with pytest.raises(ConfigError, match="형식"):
        from_dict({"transducers": [{"center": "abc"}]})


def test_presets_valid_and_array_steering():
    """모든 예시가 격자화되고, 4소자 배열 (위상차 +90°, 간격 λ/2) 빔이 +y 쪽 약 30° (= asin(Δφ/(kd))) 로 꺾인다."""
    from polarsim.engine3d.config3d import presets

    for p in presets().values():
        setup_grid(p.config)
    cfg = presets()["array_steer"].config
    r = run(cfg)
    X, Y, Z = np.meshgrid(r.xc, r.yc, r.zc, indexing="ij")
    d = np.stack([X + 0.6, Y, Z], -1)
    rr = np.linalg.norm(d, axis=-1)
    ring = (rr > 1.2) & (rr < 1.4) & (np.abs(Z) < 0.05)
    k = np.argmax(np.where(ring, np.abs(r.P), 0))
    ang = math.degrees(math.atan2(d.reshape(-1, 3)[k][1], d.reshape(-1, 3)[k][0]))
    assert ang == pytest.approx(30.0, abs=6.0)  # 소자 4개 + 근거리 (r ≈ 4λ) 라 이론값과 조금 다름
