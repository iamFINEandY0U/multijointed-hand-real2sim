"""
Step 2 - CMA-ES 파라미터 튜닝 (Isaac Sim 비의존 순수 로직).

이 모듈은 numpy와 vendor된 `cma`(pycma)만 쓴다. Isaac Sim 런타임에 의존하는
부분(실제 rollout = 트윈에 게인 세팅 후 궤적 재생 + loss 계산)은 호출자가
`evaluate_fn` 콜백으로 주입한다 — 단일 env는 후보를 순차로, 병렬 env는 physX
cloner로 배치 실행. 덕분에 이 파일은 sandbox에서 가짜 evaluate_fn 으로 단위
테스트가 가능하다 (trajectory_io.py 와 같은 정책).

튜닝 대상:
    16개 관절 각각의 (stiffness, damping)  -> 파라미터 벡터 차원 = 32.
    ("Kp"는 Isaac Sim / PhysX drive 용어로 stiffness 와 같은 값이다.)

탐색 방식:
    - 기본은 **로그 공간** 최적화: y = log10(g + shift) 를 CMA-ES 변수로 쓰고
      g = 10**y - shift 로 되돌린다. shift(>0) 덕분에 damping 초기값 0 도
      log10(0)=-inf 없이 표현된다. 게인이 초기(공장)값에서 100~10000배
      떨어져 있어도 한 번의 실행으로 도달할 수 있게 경계를 넉넉히 잡는다
      (상한 = max(|g0|*log_range_factor, 절대상한)).
    - use_log_space=False 면 예전 선형 박스([g0/factor, g0*factor], g0==0 이면
      [0, *_hi_when_zero])를 그대로 쓴다.
    - 관절마다 게인 크기가 크게 다를 수 있으므로 좌표별 초기 스텝을
      CMA_stds(= 허용 범위의 일정 비율)로 스케일해서 조건수 문제를 완화한다.
    - 궤적 loss 에 물리 타당성 페널티(effort prior + damping-ratio prior)를
      더해 "거대 stiffness + 거의 0 damping" 퇴화 해를 억제한다
      (regularization_penalty; weight=0 이면 완전 비활성).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import numpy as np

LOSS_KINDS = ("MSE", "RMSE", "MAE")


def _import_cma():
    """익스텐션에 vendor된 pycma를 import. (없으면 명확한 에러)

    _vendor/ 를 sys.path 앞에 넣어 `import cma` 가 vendor 사본을 잡게 한다.
    import 시 matplotlib/purecma 관련 UserWarning 이 나올 수 있어 조용히 무시.
    """
    vendor_dir = os.path.join(os.path.dirname(__file__), "_vendor")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cma  # noqa: F401  (vendored)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import the vendored cma (pycma) package. "
            f"Check that '{vendor_dir}/cma' exists. Cause: {e}"
        ) from e
    return cma


def compute_loss(q_sim: np.ndarray, q_real: np.ndarray, kind: str) -> float:
    """트윈 rollout 결과(q_sim)와 수집한 실측 궤적(q_real) 사이 손실.

    둘 다 shape (num_steps, num_joints), 같은 제어 주기·같은 관절 순서 가정.
    kind: "MSE" | "RMSE" | "MAE" (UI의 Loss indicator 값).
    """
    if kind not in LOSS_KINDS:
        raise ValueError(f"Unknown loss kind: {kind!r} (allowed: {LOSS_KINDS})")
    if q_sim.shape != q_real.shape:
        raise ValueError(
            f"q_sim{q_sim.shape} and q_real{q_real.shape} have different shapes."
        )
    diff = q_sim - q_real
    if kind == "MSE":
        return float(np.mean(diff**2))
    if kind == "RMSE":
        return float(np.sqrt(np.mean(diff**2)))
    return float(np.mean(np.abs(diff)))  # MAE


@dataclass
class TuningConfig:
    population: int = 50
    generations: int = 100
    loss_kind: str = "MSE"
    seed: int = 0
    sigma0: float = 1.0
    # 경계 = [g0/factor, g0*factor] (초기 게인 g0 > 0 일 때)
    stiffness_range_factor: float = 10.0
    damping_range_factor: float = 10.0
    # 초기 게인이 0 이라 배수로 범위를 못 잡을 때 쓰는 상한
    stiffness_hi_when_zero: float = 1000.0
    damping_hi_when_zero: float = 100.0
    # 절대 하한 (음수 게인 방지)
    stiffness_floor: float = 0.0
    damping_floor: float = 0.0
    # 좌표별 초기 스텝 = 이 비율 * (hi - lo)  [선형 공간용]
    stds_fraction: float = 0.25

    # === 로그 공간 탐색 (권장 기본값). y = log10(g + shift) 를 최적화한다.
    #     use_log_space=False 면 위의 선형 박스 로직을 그대로 쓴다. ===
    use_log_space: bool = True
    log_shift_stiffness: float = 1.0e-2   # y = log10(Kp + shift); Kp0==0 도 표현 가능
    log_shift_damping: float = 1.0e-4     # y = log10(Kd + shift); Kd0==0 도 표현 가능
    log_range_factor: float = 1000.0      # 상한 = max(|g0| * 이 값, 아래 절대 상한)
    stiffness_abs_hi: float = 1.0e5       # 로그 공간 stiffness 절대 상한
    damping_abs_hi: float = 1.0e3         # 로그 공간 damping 절대 상한
    log_stds_fraction: float = 0.20       # 좌표별 초기 스텝 = 이 비율 * (y_hi - y_lo) [decade]

    # === 물리 타당성 페널티: 궤적 loss 에 더해진다. weight=0 이면 완전 비활성. ===
    # effort: 초기(공장) 게인 대비 로그 편차 prior — 데이터가 강하게 요구하지
    #   않는 한 게인이 스케일을 벗어나 폭주하지 않게 잡아준다.
    #   dev = log10((g + ref) / (g0 + ref)); ref 는 "이 이하 크기는 규제 안 함"
    #   기준 스케일이라 g0≈0(공장 게인 미설정) 이어도 폭발하지 않는다.
    effort_weight: float = 1.0e-3
    effort_ref_stiffness: float = 1.0
    effort_ref_damping: float = 1.0e-2
    # damping_ratio: zeta = Kd / (2 sqrt(Kp * I)) 가 [min, max] 밖이면 제곱 페널티 —
    #   "거대 stiffness + 거의 0 damping" 퇴화 해를 억제한다. I 를 모르면 1.0.
    damping_ratio_weight: float = 1.0e-2
    damping_ratio_min: float = 0.15
    damping_ratio_max: float = 5.0
    damping_ratio_inertia: float = 1.0

    def validate(self) -> None:
        if self.population < 2:
            raise ValueError("population must be >= 2.")
        if self.generations < 1:
            raise ValueError("generations must be >= 1.")
        if self.loss_kind not in LOSS_KINDS:
            raise ValueError(f"loss_kind must be one of {LOSS_KINDS}.")
        if self.use_log_space:
            if self.log_shift_stiffness <= 0 or self.log_shift_damping <= 0:
                raise ValueError("log_shift_* must be > 0 (log10 domain).")
            if self.log_range_factor <= 1.0:
                raise ValueError("log_range_factor must be > 1.")
        if self.effort_weight < 0 or self.damping_ratio_weight < 0:
            raise ValueError("effort_weight / damping_ratio_weight must be >= 0.")
        if self.damping_ratio_weight > 0 and not (
            0 < self.damping_ratio_min < self.damping_ratio_max
        ):
            raise ValueError("must satisfy 0 < damping_ratio_min < damping_ratio_max.")


@dataclass
class SearchSpace:
    lo: np.ndarray      # (2J,)  CMA-ES 변수 하한 [stiffness..., damping...]
    hi: np.ndarray      # (2J,)  CMA-ES 변수 상한
    stds: np.ndarray    # (2J,)  좌표별 초기 스텝
    x0: np.ndarray      # (2J,)  시작점 (경계 안쪽으로 살짝 넣음)
    num_joints: int
    log_space: bool = False
    # 로그 공간이면 g = 10**y - shift. 선형 공간이면 (2J,) 0 벡터라 변환은 항등.
    shift: "np.ndarray | None" = None

    def to_gains(self, y: np.ndarray) -> np.ndarray:
        """CMA-ES 변수 벡터 y -> 실제 게인 벡터 (2J,). 둘 다 [stiffness..., damping...]."""
        y = np.asarray(y, dtype=np.float64).ravel()
        if not self.log_space:
            return y.copy()
        g = np.power(10.0, y) - self.shift
        return np.maximum(g, 0.0)


def build_search_space(
    init_stiffness: np.ndarray, init_damping: np.ndarray, config: TuningConfig
) -> SearchSpace:
    init_stiffness = np.asarray(init_stiffness, dtype=np.float64).ravel()
    init_damping = np.asarray(init_damping, dtype=np.float64).ravel()
    if init_stiffness.shape != init_damping.shape:
        raise ValueError(
            f"init_stiffness{init_stiffness.shape} and "
            f"init_damping{init_damping.shape} have different lengths."
        )
    j = init_stiffness.size

    if config.use_log_space:
        return _build_log_search_space(init_stiffness, init_damping, config)

    def _bounds(g0, factor, floor, hi_when_zero):
        g0 = np.abs(g0)
        lo = np.where(g0 > 0, np.maximum(g0 / factor, floor), floor)
        hi = np.where(g0 > 0, g0 * factor, hi_when_zero)
        # 초기값이 아주 작아 lo==hi 가 되는 걸 방지
        hi = np.maximum(hi, lo + 1e-6)
        return lo, hi

    s_lo, s_hi = _bounds(
        init_stiffness,
        config.stiffness_range_factor,
        config.stiffness_floor,
        config.stiffness_hi_when_zero,
    )
    d_lo, d_hi = _bounds(
        init_damping,
        config.damping_range_factor,
        config.damping_floor,
        config.damping_hi_when_zero,
    )

    lo = np.concatenate([s_lo, d_lo])
    hi = np.concatenate([s_hi, d_hi])
    stds = np.maximum(config.stds_fraction * (hi - lo), 1e-9)

    x0_raw = np.concatenate([init_stiffness, init_damping])
    # 경계 위에 정확히 앉으면 pycma 경계 처리기가 경고/불안정하므로 안쪽으로 nudge
    margin = 1e-6 * (hi - lo)
    x0 = np.clip(x0_raw, lo + margin, hi - margin)

    return SearchSpace(
        lo=lo, hi=hi, stds=stds, x0=x0, num_joints=j,
        log_space=False, shift=np.zeros(2 * j),
    )


def _build_log_search_space(
    init_stiffness: np.ndarray, init_damping: np.ndarray, config: TuningConfig
) -> SearchSpace:
    """로그 공간 탐색용 SearchSpace. CMA-ES 변수 y, 게인 g = 10**y - shift.

    경계(게인 공간): [floor, max(|g0| * log_range_factor, 절대 상한)].
    shift(>0) 덕분에 g0==0(특히 damping) 도 y = log10(shift) 로 표현된다.
    """
    j = init_stiffness.size
    kp0 = np.abs(init_stiffness)
    kd0 = np.abs(init_damping)

    shift = np.concatenate([
        np.full(j, float(config.log_shift_stiffness)),
        np.full(j, float(config.log_shift_damping)),
    ])
    g0 = np.concatenate([kp0, kd0])
    g_lo = np.concatenate([
        np.full(j, float(config.stiffness_floor)),
        np.full(j, float(config.damping_floor)),
    ])
    g_hi = np.concatenate([
        np.maximum(kp0 * config.log_range_factor, config.stiffness_abs_hi),
        np.maximum(kd0 * config.log_range_factor, config.damping_abs_hi),
    ])

    y_lo = np.log10(g_lo + shift)
    y_hi = np.log10(g_hi + shift)
    y_hi = np.maximum(y_hi, y_lo + 1e-6)

    y0_raw = np.log10(np.maximum(g0, 0.0) + shift)
    margin = 1e-6 * (y_hi - y_lo)
    y0 = np.clip(y0_raw, y_lo + margin, y_hi - margin)

    stds = np.maximum(config.log_stds_fraction * (y_hi - y_lo), 1e-9)

    return SearchSpace(
        lo=y_lo, hi=y_hi, stds=stds, x0=y0, num_joints=j,
        log_space=True, shift=shift,
    )


def regularization_penalty(
    stiffness: np.ndarray,
    damping: np.ndarray,
    init_stiffness: np.ndarray,
    init_damping: np.ndarray,
    config: TuningConfig,
) -> float:
    """게인 후보에 대한 물리 타당성 페널티. 궤적 loss 에 그대로 더해진다.

    - effort: 초기(공장) 게인 대비 로그 편차의 제곱 평균 (log-Gaussian prior).
      데이터 근거 없이 게인이 스케일을 벗어나 폭주하는 걸 억제.
    - damping_ratio: zeta = Kd / (2 sqrt(Kp * I)) 가 [min, max] 를 벗어난 만큼
      제곱 페널티. "거대 stiffness + 거의 0 damping" 퇴화 해를 억제.

    두 weight 가 모두 0 이면 0.0 (기존 순수 궤적 loss 와 동일).
    """
    kp = np.abs(np.asarray(stiffness, dtype=np.float64).ravel())
    kd = np.abs(np.asarray(damping, dtype=np.float64).ravel())
    kp0 = np.abs(np.asarray(init_stiffness, dtype=np.float64).ravel())
    kd0 = np.abs(np.asarray(init_damping, dtype=np.float64).ravel())

    penalty = 0.0

    if config.effort_weight > 0.0:
        rs = config.effort_ref_stiffness
        rd = config.effort_ref_damping
        dev_kp = np.log10((kp + rs) / (kp0 + rs))
        dev_kd = np.log10((kd + rd) / (kd0 + rd))
        penalty += config.effort_weight * float(
            np.mean(dev_kp ** 2) + np.mean(dev_kd ** 2)
        )

    if config.damping_ratio_weight > 0.0:
        inertia = max(config.damping_ratio_inertia, 1e-12)
        zeta = kd / (2.0 * np.sqrt(np.maximum(kp, 1e-12) * inertia))
        lo_gap = np.maximum(config.damping_ratio_min - zeta, 0.0) / config.damping_ratio_min
        hi_gap = np.maximum(zeta - config.damping_ratio_max, 0.0) / config.damping_ratio_max
        penalty += config.damping_ratio_weight * float(np.mean(lo_gap ** 2 + hi_gap ** 2))

    return penalty


@dataclass
class TuningResult:
    best_stiffness: np.ndarray
    best_damping: np.ndarray
    best_loss: float  # best-so-far 총합 loss (궤적 + 페널티)
    # best 후보의 loss 분해 (궤적 loss + 페널티 = best_loss).
    best_traj_loss: float = 0.0
    best_reg_penalty: float = 0.0
    loss_history: List[float] = field(default_factory=list)  # 세대별 best-so-far 총합 loss
    # 수렴 그래프용 세대별 통계 (loss_history 와 길이·순서 동일).
    traj_loss_history: List[float] = field(default_factory=list)    # best 후보의 궤적 loss
    reg_penalty_history: List[float] = field(default_factory=list)  # best 후보의 페널티
    gen_min_history: List[float] = field(default_factory=list)     # 그 세대 후보 중 최소 총합 loss
    gen_median_history: List[float] = field(default_factory=list)  # 그 세대 후보 총합 loss 중앙값
    sigma_history: List[float] = field(default_factory=list)       # 세대 종료 시 CMA-ES step-size σ
    generations_run: int = 0
    stopped_early: bool = False  # 사용자가 should_stop 으로 중단
    cma_stop_reason: str = ""  # CMA-ES 자체 종료 조건(수렴 등)으로 멈춘 경우


# 한 후보 = (stiffness (L,), damping (L,))  — L = 튜닝 대상 관절 수
Candidate = "tuple[np.ndarray, np.ndarray]"
# evaluate_fn(candidates: list[Candidate]) -> awaitable[list[float]]
#   후보 리스트를 받아 같은 순서로 loss 리스트를 돌려준다. rollout(트윈 재생)과
#   compute_loss 는 이 콜백 안에서 처리한다 — 단일 env는 후보를 순차로, 병렬 env는
#   physX cloner로 배치 실행할 수 있게 CMA-ES 루프와 분리했다.
EvaluateFn = Callable[[list], Awaitable[List[float]]]
# progress_cb(generation:int, best_loss:float, fraction:float) -> None
ProgressCb = Callable[[int, float, float], None]


async def run_cma_tuning(
    evaluate_fn: EvaluateFn,
    init_stiffness: np.ndarray,
    init_damping: np.ndarray,
    config: TuningConfig,
    progress_cb: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TuningResult:
    """CMA-ES ask/tell 루프.

    매 세대:
      1. es.ask()  -> 파라미터 후보 population 개 (경계 안에서 샘플)
      2. evaluate_fn(후보들) -> 각 후보의 loss (트윈 rollout + loss 계산은 콜백이 담당)
      3. es.tell(후보들, 손실들) -> 분포(m, sigma, C) 갱신
      4. progress_cb 로 UI 갱신, should_stop() 이 True 면 조기 종료

    init_stiffness/init_damping 은 튜닝 대상 관절만 담은 길이 L 배열. 반환하는
    best_stiffness/best_damping 도 길이 L (호출자가 전체 관절 배열로 재구성).
    """
    config.validate()
    cma = _import_cma()

    init_stiffness = np.asarray(init_stiffness, dtype=np.float64).ravel()
    init_damping = np.asarray(init_damping, dtype=np.float64).ravel()
    j = init_stiffness.size

    space = build_search_space(init_stiffness, init_damping, config)

    es = cma.CMAEvolutionStrategy(
        space.x0.tolist(),
        config.sigma0,
        {
            "bounds": [space.lo.tolist(), space.hi.tolist()],
            "popsize": config.population,
            "CMA_stds": space.stds.tolist(),
            "seed": config.seed,
            "maxiter": config.generations,
            "verbose": -9,
        },
    )

    best_loss = float("inf")
    best_traj_loss = float("inf")
    best_reg_penalty = 0.0
    best_stiffness = init_stiffness.copy()
    best_damping = init_damping.copy()
    loss_history: List[float] = []
    traj_loss_history: List[float] = []
    reg_penalty_history: List[float] = []
    gen_min_history: List[float] = []
    gen_median_history: List[float] = []
    sigma_history: List[float] = []
    stopped_early = False
    cma_stop_reason = ""
    gen = 0

    while gen < config.generations:
        if should_stop is not None and should_stop():
            stopped_early = True
            break

        solutions = es.ask()
        # CMA-ES 변수 -> 실제 게인 (로그 공간이면 g = 10**y - shift).
        candidates = []
        for x in solutions:
            g = space.to_gains(x)
            candidates.append((g[:j], g[j:]))

        raw_losses = await evaluate_fn(candidates)
        if len(raw_losses) != len(candidates):
            raise RuntimeError(
                f"evaluate_fn returned {len(raw_losses)} losses for {len(candidates)} candidates "
                "(count and order must match)."
            )
        # NaN/inf rollout(발산 등)은 큰 페널티로 대체해 CMA-ES가 피하게 함
        traj_losses = [
            float(v) if np.isfinite(v) else 1e12 for v in (float(x) for x in raw_losses)
        ]
        # 물리 타당성 페널티 (weight=0 이면 전부 0.0)
        reg_pens = [
            regularization_penalty(cs, cd, init_stiffness, init_damping, config)
            for cs, cd in candidates
        ]
        losses = [t + r for t, r in zip(traj_losses, reg_pens)]

        es.tell(solutions, losses)
        gen += 1

        gi = int(np.argmin(losses))
        if losses[gi] < best_loss:
            best_loss = losses[gi]
            best_traj_loss = traj_losses[gi]
            best_reg_penalty = reg_pens[gi]
            best_stiffness = candidates[gi][0].copy()
            best_damping = candidates[gi][1].copy()
        loss_history.append(best_loss)
        traj_loss_history.append(best_traj_loss)
        reg_penalty_history.append(best_reg_penalty)
        gen_min_history.append(float(np.min(losses)))
        gen_median_history.append(float(np.median(losses)))
        sigma_history.append(float(getattr(es, "sigma", float("nan"))))

        if progress_cb is not None:
            progress_cb(gen, best_loss, gen / config.generations)

        stop_dict = es.stop()
        if stop_dict:
            # CMA-ES가 자체 종료 조건에 도달 (수렴/평탄 등). 남은 세대는 건너뛰되
            # 이유를 남겨 호출자가 로그/상태로 보여줄 수 있게 한다.
            cma_stop_reason = ", ".join(f"{k}={v}" for k, v in stop_dict.items())
            break

    return TuningResult(
        best_stiffness=best_stiffness,
        best_damping=best_damping,
        best_loss=best_loss,
        best_traj_loss=best_traj_loss,
        best_reg_penalty=best_reg_penalty,
        loss_history=loss_history,
        traj_loss_history=traj_loss_history,
        reg_penalty_history=reg_penalty_history,
        gen_min_history=gen_min_history,
        gen_median_history=gen_median_history,
        sigma_history=sigma_history,
        generations_run=gen,
        stopped_early=stopped_early,
        cma_stop_reason=cma_stop_reason,
    )
