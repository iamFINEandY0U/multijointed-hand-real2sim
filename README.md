# Twin Param Tuner

**다관절 로봇 핸드의 Real2Sim 파라미터 캘리브레이션을 위한 Isaac Sim Extension**
(*Isaac Sim Extension for Real2Sim Parameter Calibration of Multi-Jointed Robot Hands*)

실제 로봇 핸드(hardware)와 Isaac Sim 트윈(digital twin) 사이의 거동 차이를 줄이기 위해,
궤적 데이터 수집 → 파라미터 최적화 → 검증을 하나의 마법사(wizard)형 도구로 통합한
Omniverse Kit / Isaac Sim Extension입니다. 관절별 PhysX 드라이브 파라미터(stiffness,
damping)를 CMA-ES(진화 전략)로 자동 정합(calibration)합니다.

## 배경

로봇 핸드는 관절마다 강성(stiffness)·감쇠(damping) 등 여러 물리 파라미터를 가져,
실물과 가상의 거동을 일치시키기 위해 조정해야 할 파라미터 수가 많습니다. 이를 사람이
수작업으로 조정하는 방식은 시간이 오래 걸리고 재현성이 떨어지며, 통상적인 절차(궤적
수집 → 별도 스크립트로 오프라인 최적화)는 데이터 수집·최적화·검증 단계가 분리되어
작업 흐름이 파편화됩니다. 이 Extension은 이 세 단계를 하나의 도구로 통합해 위 문제를
해결합니다.

## 파이프라인

세 단계로 구성되며, 데이터 수집 단계는 실물 유무에 따라 두 모드로 분기한 뒤 궤적 쌍을
수집하는 지점에서 합류합니다. 실물과의 통신은 데이터 수집 단계에서만 필요하며, 이후
튜닝·검증 단계는 시뮬레이션 내부에서만 수행되므로 하드웨어 없이 기록된 궤적만으로도
전체 절차를 재현할 수 있습니다.

1. **Data collection (Step 1)** — 사전 정의한 기준 궤적(`q_ref`, 관절별 진폭/오프셋을
   재매핑한 정현파)을 트윈에 재생해 `q_sim`을 수집합니다.
   - *실물 제어 모드*: 같은 기준 궤적을 실제 하드웨어에도 재생해 매 제어 주기마다
     피드백을 측정, `q_real`을 함께 수집합니다.
   - *파일 로드 모드*: 하드웨어 없이 이미 기록된 기준/피드백 궤적 CSV를 직접 불러옵니다.
2. **Parameter tuning (Step 2)** — CMA-ES(Covariance Matrix Adaptation Evolution
   Strategy)로 관절별 (stiffness, damping)을 정합합니다. 관절 수 N에 대해 학습 변수는
   2N개이며, 매 세대 표본화된 후보들은 PhysX 환경 복제(cloner)로 한 장면에 여러 개
   복제된 핸드에 각각 주입되어 배치 단위로 병렬 rollout·평가됩니다.
3. **Verification (Step 3)** — 최적 파라미터를 트윈에 적용(`Apply to twin`, USD 저장)한
   뒤 정합 전/후 궤적을 비교(`Matching results`: match score, mean/max error, 관절별
   오차, 시계열 그래프)하여 정합 성능을 확인합니다.

### 손실 함수

궤적 손실은 관절별 실측(`q_real`)·시뮬레이션(`q_sim`) 궤적 오차의 평균제곱오차(MSE)로
정의합니다.

```
L_traj = (1/N) * Σ_k MSE(q_real_k, q_sim_k)
```

CMA-ES가 실제로 최소화하는 목적함수는 여기에 물리적 타당성을 위한 정규화 항을
더한 것입니다.

```
L = L_traj + λ1 * R_prior + λ2 * P_damp
```

- `R_prior`: 후보 게인이 초기(공장) 게인 `(kp0, kd0)`에서 로그 스케일로 벗어난 정도에
  대한 prior.
- `P_damp`: 감쇠비 `ζ = kd / (2·sqrt(kp·I))` (관성 `I`는 미지수이므로 1로 가정)가
  사전에 정한 범위를 벗어난 정도에 대한 penalty.
- 기본 가중치: `λ1 = 1×10⁻³`, `λ2 = 1×10⁻²`.

## 적용 사례 — Delto Gripper (DG-3F-B)

12관절 Delto Gripper(DG-3F-B, 정합 대상 24개 파라미터)에 적용해 검증했습니다.
제어/측정 주기 15 Hz, 측정 길이 3 s 기준 CMA-ES 설정:

| 항목 | 값 |
|---|---|
| Parameters | stiffness ×12, damping ×12 |
| Stiffness range (N·m/deg) | [0, max(kp0×1000, 1745)] |
| Damping range (N·m·s/deg) | [0, max(kd0×1000, 17.5)] |
| Population size | 48 |
| Generations | 200 |
| Parallel envs (PhysX) | 24 (세대당 24개씩 2회 순차 배치) |

전체 튜닝 소요 시간은 약 107.4초였으며, 12관절 전체 평균제곱오차(MSE) 기준
약 65.2%의 정합 성능 개선을 확인했습니다.

## 요구 사항

- Isaac Sim / Omniverse Kit (검증 환경: Isaac Sim 5.1, PhysX 백엔드)
- Python (Isaac Sim 내장 인터프리터), `numpy`
- CMA-ES 구현은 별도 설치 없이 `twin_param_tuner/_vendor/`에 vendor되어 있습니다
  (pycma 4.4.4, BSD-3-Clause).

## 설치

1. 이 저장소(또는 이 폴더의 부모 디렉터리)를 Isaac Sim의 Extension search path에 추가합니다.
   - Window → Extensions → ⚙ (설정) → Extension Search Paths → 부모 경로 추가
2. Extensions 목록에서 `twin_param_tuner`를 검색해 활성화합니다.

## 경로 설정 (필수)

개인 개발 환경 경로(사용자 계정명 등)가 저장소에 포함되지 않도록, 아래 기본 경로들은
`twin_param_tuner/extension.py`에 빈 문자열(`""`)로 되어 있습니다. **사용자 환경에 맞게
직접 채워 넣어야 정상 동작합니다.**

| 변수 | 용도 |
|---|---|
| `_default_save_dir` | 결과 CSV(q_sim.csv, q_real.csv, tuned_gains.csv 등)를 저장할 폴더 |
| `_default_hw_script_path` | 실물 하드웨어 replay 스크립트(`ReplayRefTrajectory.py`) 경로 (실물 제어 모드에서만 필요) |
| `_default_hw_runner_cmd` | 위 스크립트를 실행할 Python 인터프리터 경로 (실물 제어 모드에서만 필요) |

빈 값으로 두어도 Extension은 실행되며, UI에서 매번 직접 경로를 입력해도 됩니다.
반복 입력이 번거로우면 위 세 변수를 자신의 로컬 경로로 채워 넣으세요 (이 파일은 저장소에
공유되므로, 실제 개인 경로로 채운 뒤에는 커밋 전에 다시 지우거나 로컬 전용 설정으로
분리하는 것을 권장합니다).

## 사용법

1. **Step 1 — Data collection**: Robot Selection에서 트윈 articulation prim을 지정하고,
   `q_ref` CSV를 로드한 뒤 Control cycle / Measurement length를 궤적에 맞게 설정하고
   Data collection을 실행합니다.
2. **Step 2 — Tuning**: `q_real` CSV(또는 Step 1에서 자동 채워진 값)를 확인하고,
   Population / Generations / Loss indicator / Learning parameters / Env(병렬 환경 수)를
   설정한 뒤 Tuning start로 CMA-ES 튜닝을 실행합니다.
3. **Step 3 — Verification**: 튜닝이 끝나면 Apply to twin으로 결과 게인을 적용하고,
   Matching results로 적용 전/후 오차를 비교합니다.

## 프로젝트 구조

```
twin_param_tuner/
  config/
    extension.toml        # Extension 메타데이터(title, version, dependencies)
  twin_param_tuner/
    __init__.py
    extension.py           # UI 및 Step 1/2/3 배선
    trajectory_io.py        # CSV 입출력, Trajectory 데이터 구조, 검증
    twin_playback.py        # 트윈 rollout 실행기(단일/병렬 환경), USD 드라이브 적용
    cma_tuning.py            # CMA-ES 튜닝 루프(순수 로직, Isaac Sim 비의존)
    _vendor/cma/             # vendor된 pycma (BSD-3-Clause)
  tests/
    test_cma_tuning.py
    test_twin_playback.py
```

## 테스트

Isaac Sim 비의존 로직(`cma_tuning.py`)은 일반 Python 환경에서 pytest로 실행할 수 있습니다.

```bash
pytest twin_param_tuner/tests/test_cma_tuning.py
```

## Acknowledgement

한국전자기술연구원(KETI) · 연구개발과제(No. RS-2024-00417108)

## 라이선스

`twin_param_tuner/_vendor/cma/`에 포함된 pycma는 BSD 3-Clause 라이선스를 따릅니다.
자세한 내용은 `twin_param_tuner/_vendor/README.md`를 참고하세요.
