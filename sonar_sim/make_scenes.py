"""예제 장면 JSON 생성기 (scenes/*.json 을 코드로 만들어 키 오타를 방지). 실행: python make_scenes.py"""
from dataclasses import replace
from pathlib import Path

from sss_sim.config import ObjectSpec, SceneConfig

OUT = Path(__file__).resolve().parent / "scenes"


def scenes():
    base = SceneConfig()
    yield replace(base, name="default", objects=[
        ObjectSpec("cylinder", 12.0, 14.0, 2.5, 0.6, heading_deg=20.0, name="cylinder_mine"),
        ObjectSpec("sphere", 22.0, -10.0, width=0.9, name="moored_sphere"),
        ObjectSpec("box", 30.0, -24.0, 2.0, 1.5, 1.0, heading_deg=-30.0, name="crate"),
        ObjectSpec("cylinder", 33.0, 26.0, 1.5, 0.5, heading_deg=95.0, burial_m=0.2, name="half_buried"),
    ])
    # 난파선 현장: 상자 여러 개로 선체/상부구조, 주변 잔해, 거친 자갈 해저
    hull = [ObjectSpec("box", 25.0, 18.0, 12.0, 3.0, 1.6, heading_deg=35.0, reflectivity=5.0, name="hull"),
            ObjectSpec("box", 24.0, 17.3, 4.0, 2.0, 2.8, heading_deg=35.0, reflectivity=5.0, name="deckhouse"),
            ObjectSpec("cylinder", 30.0, 21.5, 3.0, 0.5, heading_deg=35.0, name="mast")]
    debris = [ObjectSpec("box", x, y, 0.8, 0.6, 0.4, heading_deg=h, reflectivity=4.0, specular=15.0, name="debris")
              for x, y, h in ((19.0, 10.0, 10), (21.5, 9.0, 70), (33.0, 14.0, 120), (16.0, 22.0, 40))]
    yield replace(base, name="wreck_site", seed=3, trajectory=replace(base.trajectory, end_xy=(50.0, 0.0)),
                  seabed=replace(base.seabed, gravel_fraction=0.35, n_rocks=40, roughness_rms_m=0.12, ripple_coverage=0.15),
                  objects=hull + debris)
    # 얕은 바다 파이프라인: 수면 반사/다중경로가 잘 보이는 조건
    yield replace(base, name="pipeline_shallow", seed=5,
                  sonar=replace(base.sonar, sidelobe_db=-18.0),
                  trajectory=replace(base.trajectory, altitude_m=6.0, water_depth_m=9.0),
                  seabed=replace(base.seabed, ripple_coverage=0.6, ripple_amp_m=0.04, mud_fraction=0.4),
                  noise=replace(base.noise, surface_return_db=3.0, multipath_db=-10.0),
                  objects=[ObjectSpec("cylinder", 20.0, 0.0, 80.0, 0.8, heading_deg=70.0, reflectivity=4.0,
                                      specular=20.0, burial_m=0.25, name="pipeline")])
    # 열악한 조사: 큰 자세 흔들림, 낮은 SNR, 간섭, 핑 누락, 빔보정 없음
    yield replace(base, name="rough_survey", seed=11,
                  sonar=replace(base.sonar, beam_correction=False, tvg_exponent=36.0),
                  trajectory=replace(base.trajectory, roll_amp_deg=4.0, heave_amp_m=0.4, yaw_amp_deg=3.0),
                  noise=replace(base.noise, snr_far_db=6.0, interference_prob=0.02, dropout_prob=0.008, n_fish=12),
                  objects=[ObjectSpec("cylinder", 18.0, -15.0, 2.0, 0.5, heading_deg=0.0, name="target")])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for sc in scenes():
        sc.validate()
        sc.save(OUT / f"{sc.name}.json")
        print("wrote", OUT / f"{sc.name}.json")
