"""Sandbox tests for twin_param_tuner.cma_tuning (Isaac Sim 비의존 부분).

실행: `python3 tests/test_cma_tuning.py` (numpy 필요, cma 는 _vendor 에 있음).
"""
import asyncio
import importlib.util
import os
import sys

import numpy as np

# extension.py 는 carb/omni 를 import 하므로 패키지 __init__ 을 거치지 않고
# cma_tuning.py 만 직접 로드한다 (순수 로직 모듈).
_CMA_TUNING_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "twin_param_tuner",
    "cma_tuning.py",
)
_spec = importlib.util.spec_from_file_location("cma_tuning", _CMA_TUNING_PY)
cma_tuning = importlib.util.module_from_spec(_spec)
sys.modules["cma_tuning"] = cma_tuning
_spec.loader.exec_module(cma_tuning)

TuningConfig = cma_tuning.TuningConfig
build_search_space = cma_tuning.build_search_space
compute_loss = cma_tuning.compute_loss
run_cma_tuning = cma_tuning.run_cma_tuning
regularization_penalty = cma_tuning.regularization_penalty

# 옵티마이저 메커니즘만 보는 테스트용 config: 선형 박스 + 페널티 없음
# (로그 공간/페널티는 별도 테스트에서 다룬다).
_MECH = dict(use_log_space=False, effort_weight=0.0, damping_ratio_weight=0.0)


def test_compute_loss():
    q_real = np.zeros((5, 3))
    q_sim = np.full((5, 3), 2.0)
    assert abs(compute_loss(q_sim, q_real, "MSE") - 4.0) < 1e-12
    assert abs(compute_loss(q_sim, q_real, "RMSE") - 2.0) < 1e-12
    assert abs(compute_loss(q_sim, q_real, "MAE") - 2.0) < 1e-12
    try:
        compute_loss(q_sim, q_real, "BOGUS")
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        compute_loss(np.zeros((4, 3)), q_real, "MSE")
        assert False, "shape mismatch should raise"
    except ValueError:
        pass
    print("test_compute_loss OK")


def test_build_search_space_zero_damping():
    init_s = np.array([100.0, 50.0, 200.0, 10.0])
    init_d = np.array([0.0, 0.0, 2.0, 0.0])
    cfg = TuningConfig(stiffness_range_factor=10.0, damping_range_factor=10.0, **_MECH)
    sp = build_search_space(init_s, init_d, cfg)
    assert sp.num_joints == 4
    assert not sp.log_space
    # stiffness lower = s/10, upper = s*10
    assert np.allclose(sp.lo[:4], init_s / 10.0)
    assert np.allclose(sp.hi[:4], init_s * 10.0)
    # damping==0 -> [0, hi_when_zero]
    assert np.allclose(sp.lo[4:][init_d == 0], 0.0)
    assert np.allclose(sp.hi[4:][init_d == 0], cfg.damping_hi_when_zero)
    # damping==2 -> [0.2, 20]
    assert abs(sp.lo[4 + 2] - 0.2) < 1e-9
    assert abs(sp.hi[4 + 2] - 20.0) < 1e-9
    # x0 strictly inside bounds
    assert np.all(sp.x0 > sp.lo) and np.all(sp.x0 < sp.hi)
    assert np.all(sp.stds > 0)
    print("test_build_search_space_zero_damping OK")


def test_run_cma_tuning_converges():
    rng = np.random.default_rng(0)
    j = 4
    n_steps = 60
    q_real = rng.standard_normal((n_steps, j))

    s_true = np.array([120.0, 60.0, 240.0, 12.0])
    d_true = np.array([1.5, 0.8, 3.0, 0.4])
    a = np.array([0.01, 0.02, 0.005, 0.05])
    b = np.array([0.2, 0.3, 0.1, 0.5])
    # 두 게인이 서로 독립적으로 식별되도록 시간 프로파일을 다르게 준다:
    # stiffness 항은 상수(f_t=1), damping 항은 램프(g_t=t/(N-1)).
    f_t = np.ones(n_steps)[:, None]
    g_t = (np.arange(n_steps) / (n_steps - 1))[:, None]

    def _rollout(stiffness, damping):
        ds = (a * (stiffness - s_true))[None, :]  # (1, j)
        dd = (b * (damping - d_true))[None, :]
        return q_real + ds * f_t + dd * g_t

    async def evaluate_fn(candidates):
        return [compute_loss(_rollout(s, d), q_real, "MSE") for s, d in candidates]

    init_s = s_true * 1.6
    init_d = d_true * 1.6
    init_loss = compute_loss(_rollout(init_s, init_d), q_real, "MSE")

    cfg = TuningConfig(population=16, generations=80, loss_kind="MSE", seed=1, **_MECH)
    result = asyncio.new_event_loop().run_until_complete(
        run_cma_tuning(evaluate_fn, init_s, init_d, cfg)
    )
    print(f"  init MSE loss   = {init_loss:.6g}")
    print(f"  best MSE loss   = {result.best_loss:.6g}")
    print(f"  generations_run = {result.generations_run}")
    print(f"  best stiffness  = {np.round(result.best_stiffness, 2)} (true {s_true})")
    print(f"  best damping    = {np.round(result.best_damping, 3)} (true {d_true})")
    assert result.generations_run == 80
    assert len(result.loss_history) == 80
    assert result.best_loss < init_loss * 0.05, "CMA-ES should cut loss by >20x"
    assert np.allclose(result.best_stiffness, s_true, rtol=0.15, atol=2.0)
    assert np.allclose(result.best_damping, d_true, rtol=0.2, atol=0.2)
    # monotone non-increasing best-loss history
    assert all(x >= y - 1e-12 for x, y in zip(result.loss_history, result.loss_history[1:]))
    # 수렴 그래프용 세대별 통계도 같은 길이로 채워져야 한다.
    assert len(result.gen_min_history) == 80
    assert len(result.gen_median_history) == 80
    assert len(result.sigma_history) == 80
    assert len(result.traj_loss_history) == 80
    assert len(result.reg_penalty_history) == 80
    # 페널티 꺼둔 config 이므로 reg 는 전부 0, best_loss == best_traj_loss
    assert all(r == 0.0 for r in result.reg_penalty_history)
    assert abs(result.best_loss - result.best_traj_loss) < 1e-12
    assert result.best_reg_penalty == 0.0
    # best-so-far <= 그 세대 최소 <= 그 세대 중앙값
    assert all(b <= m + 1e-12 for b, m in zip(result.loss_history, result.gen_min_history))
    assert all(
        mn <= md + 1e-12
        for mn, md in zip(result.gen_min_history, result.gen_median_history)
    )
    print("test_run_cma_tuning_converges OK")


def test_should_stop_early():
    q_real = np.zeros((10, 2))
    ramp = (np.arange(10) / 9.0)[:, None]

    async def evaluate_fn(candidates):
        # 비평탄 목적함수라 CMA-ES es.stop() 이 먼저 안 터지게 함
        out = []
        for stiffness, damping in candidates:
            ds = (0.01 * (stiffness - 50.0))[None, :]
            dd = (0.1 * (damping - 5.0))[None, :]
            out.append(compute_loss(q_real + ds + dd * ramp, q_real, "RMSE"))
        return out

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 3  # stop after a few generations

    cfg = TuningConfig(population=8, generations=100, loss_kind="RMSE", seed=2, **_MECH)
    result = asyncio.new_event_loop().run_until_complete(
        run_cma_tuning(evaluate_fn, np.array([10.0, 10.0]), np.array([1.0, 1.0]),
                       cfg, should_stop=should_stop)
    )
    assert result.stopped_early
    assert result.generations_run < 100
    print(f"test_should_stop_early OK (ran {result.generations_run} gens)")


def test_batched_evaluate_like_parallel_envs():
    """evaluate_fn 이 후보를 env_count 크기로 잘라 배치 평가해도 (병렬 env 흉내)
    순서가 유지되고 수렴한다."""
    rng = np.random.default_rng(3)
    j = 3
    n_steps = 40
    q_real = rng.standard_normal((n_steps, j))
    s_true = np.array([80.0, 150.0, 20.0])
    d_true = np.array([2.0, 1.0, 0.5])
    a = np.array([0.02, 0.01, 0.04])
    b = np.array([0.25, 0.15, 0.4])
    f_t = np.ones(n_steps)[:, None]
    g_t = (np.arange(n_steps) / (n_steps - 1))[:, None]
    env_count = 5

    def _rollout(s, d):
        return q_real + (a * (s - s_true))[None, :] * f_t + (b * (d - d_true))[None, :] * g_t

    seen_chunk_sizes = []

    async def evaluate_fn(candidates):
        losses = []
        for start in range(0, len(candidates), env_count):
            chunk = candidates[start : start + env_count]
            seen_chunk_sizes.append(len(chunk))
            # 배치로 한꺼번에 계산 (병렬 env 흉내)
            batch = np.stack([_rollout(s, d) for s, d in chunk])  # (P, n_steps, j)
            for k in range(len(chunk)):
                losses.append(compute_loss(batch[k], q_real, "MSE"))
        return losses

    cfg = TuningConfig(population=16, generations=60, loss_kind="MSE", seed=7, **_MECH)
    result = asyncio.new_event_loop().run_until_complete(
        run_cma_tuning(evaluate_fn, s_true * 1.5, d_true * 1.5, cfg)
    )
    assert set(seen_chunk_sizes[:4]) == {5, 1} or seen_chunk_sizes[:4] == [5, 5, 5, 1]
    assert result.best_loss < 1e-3
    assert np.allclose(result.best_stiffness, s_true, rtol=0.15, atol=3.0)
    assert np.allclose(result.best_damping, d_true, rtol=0.25, atol=0.25)
    print(f"test_batched_evaluate_like_parallel_envs OK (best MSE {result.best_loss:.3g})")


def test_build_log_search_space():
    init_s = np.array([100.0, 50.0, 10.0])
    init_d = np.array([0.0, 2.0, 0.0])  # damping 0 도 표현돼야 함
    cfg = TuningConfig()  # 기본 = 로그 공간
    sp = build_search_space(init_s, init_d, cfg)
    assert sp.log_space and sp.shift is not None
    assert sp.lo.shape == sp.hi.shape == sp.x0.shape == (6,)
    # y 는 유한, x0 는 경계 안쪽
    assert np.all(np.isfinite(sp.lo)) and np.all(np.isfinite(sp.hi))
    assert np.all(sp.x0 > sp.lo) and np.all(sp.x0 < sp.hi)
    # x0 를 게인으로 되돌리면 초기 게인과 일치 (round-trip)
    g0 = sp.to_gains(sp.x0)
    assert np.allclose(g0[:3], init_s, rtol=1e-6, atol=1e-6)
    assert np.allclose(g0[3:], init_d, rtol=1e-6, atol=1e-4)
    # 하한을 게인으로 되돌리면 floor(0), 상한은 절대 상한 이상
    g_lo = sp.to_gains(sp.lo)
    assert np.allclose(g_lo, 0.0, atol=1e-6)
    g_hi = sp.to_gains(sp.hi)
    assert g_hi[0] >= cfg.stiffness_abs_hi - 1.0   # 100 * 1000 = 1e5 == abs_hi
    assert g_hi[3] >= cfg.damping_abs_hi - 1.0     # init 0 -> abs_hi
    print("test_build_log_search_space OK")


def test_log_space_reaches_far_optimum():
    """로그 공간이면 초기값에서 100~1000배 떨어진 최적 게인도 한 번에 도달."""
    rng = np.random.default_rng(11)
    j = 3
    n_steps = 50
    q_real = rng.standard_normal((n_steps, j))
    s_true = np.array([5000.0, 20.0, 800.0])   # init 대비 최대 ~250x
    d_true = np.array([50.0, 0.3, 8.0])
    a = np.array([2e-4, 5e-3, 1e-3])
    b = np.array([5e-3, 0.2, 3e-2])
    f_t = np.ones(n_steps)[:, None]
    g_t = (np.arange(n_steps) / (n_steps - 1))[:, None]

    def _rollout(s, d):
        return q_real + (a * (s - s_true))[None, :] * f_t + (b * (d - d_true))[None, :] * g_t

    async def evaluate_fn(candidates):
        return [compute_loss(_rollout(s, d), q_real, "MSE") for s, d in candidates]

    init_s = np.array([20.0, 20.0, 20.0])
    init_d = np.array([0.2, 0.2, 0.2])
    init_loss = compute_loss(_rollout(init_s, init_d), q_real, "MSE")

    # 로그 공간 ON, 페널티는 이 테스트에서 방해되지 않게 OFF
    cfg = TuningConfig(
        population=20, generations=250, loss_kind="MSE", seed=3,
        effort_weight=0.0, damping_ratio_weight=0.0,
    )
    result = asyncio.new_event_loop().run_until_complete(
        run_cma_tuning(evaluate_fn, init_s, init_d, cfg)
    )
    print(f"  init loss={init_loss:.4g} -> best={result.best_loss:.4g}")
    print(f"  best s={np.round(result.best_stiffness,1)} (true {s_true})")
    print(f"  best d={np.round(result.best_damping,3)} (true {d_true})")
    assert result.best_loss < init_loss * 0.02
    assert np.allclose(result.best_stiffness, s_true, rtol=0.15, atol=5.0)
    assert np.allclose(result.best_damping, d_true, rtol=0.2, atol=0.2)
    print("test_log_space_reaches_far_optimum OK")


def test_regularization_penalty():
    cfg0 = TuningConfig(effort_weight=0.0, damping_ratio_weight=0.0)
    s0 = np.array([100.0, 100.0]); d0 = np.array([1.0, 1.0])
    # weight 0 이면 항상 0
    assert regularization_penalty(s0 * 1000, d0 * 0.0, s0, d0, cfg0) == 0.0

    # effort: 초기값과 같으면 ~0, 로그로 멀어질수록 커짐
    cfg_e = TuningConfig(effort_weight=1.0, damping_ratio_weight=0.0)
    assert regularization_penalty(s0, d0, s0, d0, cfg_e) < 1e-6
    near = regularization_penalty(s0 * 2, d0 * 2, s0, d0, cfg_e)
    far = regularization_penalty(s0 * 100, d0 * 100, s0, d0, cfg_e)
    assert far > near > 0.0

    # damping-ratio: 거대 Kp + 거의 0 Kd 는 페널티, 정상 비율은 ~0
    cfg_z = TuningConfig(
        effort_weight=0.0, damping_ratio_weight=1.0,
        damping_ratio_min=0.15, damping_ratio_max=5.0, damping_ratio_inertia=1.0,
    )
    degen = regularization_penalty(
        np.array([1.0e5, 1.0e5]), np.array([1e-3, 1e-3]),
        np.array([100.0, 100.0]), np.array([1.0, 1.0]), cfg_z,
    )
    # zeta = Kd / (2 sqrt(Kp*I)) ; Kp=400, Kd=40 -> zeta = 40/40 = 1.0 (밴드 안)
    ok = regularization_penalty(
        np.array([400.0, 400.0]), np.array([40.0, 40.0]),
        np.array([100.0, 100.0]), np.array([1.0, 1.0]), cfg_z,
    )
    assert degen > 0.5 and ok < 1e-6
    print("test_regularization_penalty OK")


def test_regularizer_suppresses_degenerate_solution():
    """궤적 loss 는 'Kp 무한대'로 최소화되지만, damping-ratio 페널티가 켜지면
    결과 게인의 감쇠비가 밴드 근처로 유지된다."""
    j = 2
    n_steps = 40
    ramp = (np.arange(n_steps) / (n_steps - 1))[:, None]
    target = np.zeros((n_steps, j))

    def _rollout(s, d):
        # Kp 가 클수록 target 에 가까워짐 (1/Kp 로 오차 감소). d 는 궤적에 영향 X
        # -> 페널티 없으면 옵티마이저가 Kp 를 상한까지 밀어붙임.
        err = (200.0 / np.maximum(s, 1e-6))[None, :] * (1.0 + ramp)
        return target + err

    async def evaluate_fn(candidates):
        return [compute_loss(_rollout(s, d), target, "MSE") for s, d in candidates]

    init_s = np.array([50.0, 50.0])
    init_d = np.array([1.0, 1.0])

    cfg = TuningConfig(
        population=20, generations=150, loss_kind="MSE", seed=5,
        effort_weight=0.0, damping_ratio_weight=0.05,
        damping_ratio_min=0.3, damping_ratio_max=3.0, damping_ratio_inertia=1.0,
    )
    result = asyncio.new_event_loop().run_until_complete(
        run_cma_tuning(evaluate_fn, init_s, init_d, cfg)
    )
    kp = np.abs(result.best_stiffness); kd = np.abs(result.best_damping)
    zeta = kd / (2.0 * np.sqrt(np.maximum(kp, 1e-12)))
    print(f"  best Kp={np.round(kp,1)} Kd={np.round(kd,2)} zeta={np.round(zeta,2)}")
    print(f"  best_loss={result.best_loss:.4g} traj={result.best_traj_loss:.4g} "
          f"penalty={result.best_reg_penalty:.4g}")
    # 감쇠비가 붕괴(≈0)하지 않고 밴드 하한 부근 이상으로 유지돼야 함
    assert np.all(zeta > 0.15)
    assert result.best_reg_penalty >= 0.0
    print("test_regularizer_suppresses_degenerate_solution OK")


def test_evaluate_fn_length_mismatch_raises():
    async def bad_evaluate_fn(candidates):
        return [0.0] * (len(candidates) - 1)  # 하나 부족

    cfg = TuningConfig(population=6, generations=3, seed=1)
    try:
        asyncio.new_event_loop().run_until_complete(
            run_cma_tuning(bad_evaluate_fn, np.array([1.0, 1.0]), np.array([0.1, 0.1]), cfg)
        )
        assert False, "should have raised"
    except RuntimeError:
        pass
    print("test_evaluate_fn_length_mismatch_raises OK")


if __name__ == "__main__":
    test_compute_loss()
    test_build_search_space_zero_damping()
    test_build_log_search_space()
    test_run_cma_tuning_converges()
    test_log_space_reaches_far_optimum()
    test_regularization_penalty()
    test_regularizer_suppresses_degenerate_solution()
    test_should_stop_early()
    test_batched_evaluate_like_parallel_envs()
    test_evaluate_fn_length_mismatch_raises()
    print("\nALL TESTS PASSED")
