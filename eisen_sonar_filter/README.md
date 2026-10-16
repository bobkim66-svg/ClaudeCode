# eisen_sonar_filter

아이젠슈타인 정수 Z[ω]와 잉여체 지표/가우스 합을 이용한 소나 스페클 필터 **연구용 프로토타입**입니다.
실측 노이즈 영상이 없어서 물리 기반(간이) 사이드스캔 소나 시뮬레이터를 함께 제공합니다.

## 실행 (PowerShell, 이 폴더에서)

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python examples/run_denoise_demo.py            # 약 1분, results/<타임스탬프>/ 에 표·그림·CSV 저장
python examples/run_denoise_demo.py --smoke    # 빠른 점검
```

## 내 영상(sonar_sim 결과)에 필터 적용

```powershell
python filter_sonar.py "C:\...\sonar_sim\results\<타임스탬프 폴더>"     # 폴더 안 *.npz 전부 처리
python filter_sonar.py 파일.npz --methods B Lee Frost --window 7 --pfa 0.2
```
- 입력: sonar_sim 결과 `.npz` (권장) 또는 2D 선형 강도 `.npy`. PNG 는 dB/컬러맵이 입혀진 표시용이라 받지 않는다.
- 출력: `results/filtered_<타임스탬프>/` 에 비교 그림 `*_compare.png`, 방법별 영상 `*_<방법>.png`, `*_filtered.npz`, `metrics.csv`.
- 정답(`clean`)이 있으면 PSNR/SSIM/EPI 를 표시 영역(dB −18~+10 → [0,1], PNG 와 같은 스케일)에서 계산한다.
  벤치마크(`run_denoise_demo.py`)의 진폭 영역 지표와 정의가 달라서 숫자를 직접 비교하면 안 된다.

## 구조

| 파일 | 내용 |
|---|---|
| `eisen_sonar/core/eisenstein.py` | `EisensteinInt`(+, −, ×, 노름, 켤레, 유클리드 나눗셈), 좌표 변환, 직교↔육각 재샘플링(정삼각형 무게중심 보간) |
| `eisen_sonar/core/gauss_sum.py` | `GaussSumKernel`: Z[ω]/(π) ≅ F_q, χ, ψ, G(χ,ψ), 커널 가중치, 단위고리 지표 |
| `eisen_sonar/filters/gauss_filter.py` | `EisensteinGaussFilter` — `resample_only` / `linear`(A, 원안) / `adaptive`(B) |
| `eisen_sonar/filters/baseline_filters.py` | Lee, Kuan, Frost |
| `eisen_sonar/utils/data_loader.py` | SSS 시뮬레이터(`make_synthetic_sonar`), 도메인 랜덤화(`randomized_config`), 실측 로더 |
| `eisen_sonar/utils/metrics.py` | ENL, PSNR, SSIM, CNR, EPI |

## 수학

- ω = e^{2πi/3}, N(a+bω) = a² − ab + b², 격자점 (a,b) → (x,y) = (a − b/2, (√3/2)b)
- N(π) = q 가 유리소수이면 Z[ω]/(π) ≅ F_q, ω ↦ r = −a·b⁻¹ (mod q), r² + r + 1 ≡ 0
- G(χ,ψ) = Σ_{x∈F_q^*} χ(x)ψ(x): χ 비자명이면 |G| = √q, 자명이면 G = −1 (테스트로 확인)
- **핵심 사실**: q ≡ 1 (mod 3), q ≥ 7 이면 단위군 μ₆ = {1, 1+ω, ω, −1, ω², −ω} 이 F_q^* 에 단사로 들어간다.
  (q = 7 이면 동형이고, 육각 7점 클러스터가 완전잉여계다.) 따라서 χ 를 1-링 이웃으로 제한하면
  각도 조화성분 e^{imθ_j} (θ_j = 60°·j) 이 되고, 지표 직교성 Σ_j χ(ζ^j) = 0 에 의해 국소 평균(DC)은 정확히 상쇄된다.

### (A) linear — 원안 그대로
w(z) = χ(z mod π)ψ(z mod π) / Σ_z(…), 출력 = Re Σ_z w(z) I(p+z). 7점 클러스터(q=7)에서 정규화 상수가 정확히 G다.
가중치가 영상과 무관하므로 **고정 선형 FIR 필터**다. 백색잡음 분산 이득은 Σ(Re w)² = 2/7 이다 (해석적으로 계산했고 테스트로 확인; 7점 평균은 1/7).

### (B) adaptive — 지표 합을 구조 검출 통계량으로 사용
1. 로그 강도 L = log I 로 바꾼다 (곱셈 스페클 → 덧셈 잡음).
2. S_m(p) = Σ_j χ_m(ζ^j mod π) · L(p + sζ^j), m = 1(경계), 2(선)
3. 귀무(동질 영역) 가정에서 |S_m|² ≈ 지수분포이므로, 영상 전체 중앙값/ln2 로 척도를 자기보정한다 → T = Σ_m |S_m|²/scale_m ~ Gamma(2,1)
4. τ = Gamma(2,1) 의 상위 p_fa 분위수, k = clip((T−τ)/τ, 0, 1)
5. 출력 = (1−k)·육각원판평균 + k·(arg S₁ + 90° 또는 arg S₂ / 2 에 가장 가까운 격자축 방향 선평균)

B의 결과는 소수 π 의 선택에 **불변**이다 (q=7 과 q=13 의 출력 차이 < 1e-12, 테스트로 확인). 따라서 π 는 튜닝 손잡이가 아니다.
또 가우스 합 G 는 B 에서 정규화 항등식 Σ_t χ(t)ψ(tx) = χ̄(x)G 로만 등장하며, 잡음 제거 효과는 지표 직교성과 μ₆ ↪ F_q^* 구조에서 나온다.

## 시뮬레이터 (sim-to-real 용)
해저 높이맵(모래 물결, 암석, 원통/상자 표적, 진흙 패치)과 재질 반사율(감마 텍스처, K-분포형 꼬리)을 쓴다.
여기에 람베르트 산란, 그림자 광선추적, 수직 빔패턴, r⁻⁴ 확산과 흡수, TVG를 적용하고 경사거리 bin 으로 누적한다.
잡음은 복소 IQ 로 만든다: Z = (√I₀·w) ⊛ PSF (공간 상관 스페클) + 거리 의존 열잡음, L-look 평균.
- 검증: 동질 패치 ENL = L (L=1: 0.94~1.04, L=4: 4.01), 평탄 해저 그림자 길이 = h·y/(H−h) 해석해 일치, 같은 시드면 같은 결과.
- `randomized_config()`: 고도, 빔 기울기, TVG 오차, SNR, 텍스처, PSF, 표적 배치를 무작위화한다 (도메인 랜덤화).

## 결과 (`docs/`, 평가 시드 20개, 튜닝 시드 5개와 분리, 단일 look)

default 시나리오 평균 (± 표준편차). 전체 내용은 `docs/summary.txt` 에 있다.

| 방법 (튜닝값) | ENL | PSNR [dB] | SSIM | EPI | CNR 표적 | CNR 그림자 | 시간 [ms] 256×512 |
|---|---|---|---|---|---|---|---|
| Noisy | 1.01 | 21.98 | 0.563 | 0.132 | 0.33 | 0.60 | – |
| Lee (5×5) | 6.12 | 27.56 | 0.813 | 0.293 | 0.52 | 0.94 | 4 |
| Kuan (5×5) | 6.42 | 27.44 | 0.811 | 0.360 | 0.58 | 0.97 | 4 |
| Frost (5×5, D=0.3) | 6.06 | 27.50 | 0.814 | **0.425** | 0.57 | 0.98 | 21 |
| HexResample (대조군) | 1.62 | 24.14 | 0.663 | 0.193 | 0.39 | 0.69 | 14 |
| A linear (R=2) | 5.24 | 26.86 | 0.790 | 0.299 | 0.56 | 0.93 | 31 |
| **B adaptive** (R=3, p_fa=0.2) | **8.58** | **27.71** | 0.812 | 0.237 | **0.60** | **1.02** | 76 |

해석 (Wilcoxon 부호순위, 쌍체, Holm 보정):
- B 는 ENL(+2.1~2.4), PSNR(+0.15~0.27 dB), 그림자 CNR 에서 Lee/Kuan/Frost 보다 유의하게 높다 (p_holm ≈ 1e−5, n=20 에서 가능한 최소값 근처).
  다만 PSNR 차이는 작고, randomized 시나리오에서는 B−Lee 의 PSNR 차이가 유의하지 않다 (p_holm = 0.08).
- SSIM 은 사실상 동률이다. **EPI(경계 보존)는 B 가 고전 필터 셋보다 모두 나쁘다** (Frost 0.43 vs B 0.24). 모래 물결 같은 약한 미세 구조를 평활하는 대가로 ENL 을 얻는다.
- A(원안)는 Lee/Kuan/Frost 보다 모든 지표에서 나쁘거나 같다.
- 육각 재샘플링 왕복만으로도 ENL 이 1.0 → 1.6 이 된다. 백색잡음 분산은 37% 로 줄어든다.

## 한계 (해석 시 주의)
- 모든 수치는 **이 시뮬레이터 안에서** 얻은 것이다 (단일 산란, 다중경로/자세 흔들림 없음, 흡수·빔폭 값은 확인 필요). 실측 영상에서는 검증하지 않았다.
- 하이퍼파라미터는 PSNR 기준으로 튜닝했다. B 는 더 큰 창(37점)을 골랐고 Lee 는 5×5(25px)를 골랐다. 창 크기를 맞춘 비교가 아니라 각자 최적값끼리의 비교다.
- 기준선 필터에는 참값 L=1 을 주었다 (oracle). B 는 L 없이 자기보정한다.
- 이 방법의 독창성과 선행연구 여부는 조사하지 않았다 (확인 필요). B 는 본질적으로 "육각 각도 조화성분 기반 Lee형 적응 필터"이고, 대수 구조는 그 계수를 주는 방식이다.
