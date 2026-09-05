"""
Mode B - Step 1: q_ref 재생 → q_sim 기록.
(q_real — 실측 정답 궤적 — 은 Step 2에서 loss 비교용으로 별도 로드한다.
 이 모듈은 q_ref/q_real 둘 다 같은 Trajectory 타입으로 다루며, 어느 쪽이든
 트윈 DOF 순서로 재배열하는 로직은 공용이다.)

주의: 이 모듈은 Isaac Sim 프로세스 안(Extension, 또는 python.sh로 띄운
standalone 스크립트)에서만 동작한다. isaacsim.* / omni.* 모듈은 Isaac Sim
런타임 밖에서는 import 자체가 안 되므로, 이 sandbox에서는 실행/테스트할
수 없다 (trajectory_io.py 는 순수 Python이라 sandbox에서 실행 검증함).

사용한 API는 Isaac Sim 5.0/5.1 공식 문서 기준으로 확인한 것들이다:
  - isaacsim.core.api.SimulationContext        (physics_dt / rendering_dt 제어)
  - isaacsim.core.prims.SingleArticulation     (단일 Articulation 래퍼)
  - isaacsim.core.utils.types.ArticulationAction

Isaac Sim 버전에 따라 클래스 위치가 바뀔 수 있으므로(4.x는 omni.isaac.core,
5.x는 isaacsim.core.*), 실제 환경 버전에서 import 가 실패하면 그 버전의
문서를 다시 확인해야 한다 — 여기서 임의로 다른 이름을 지어내지 않는다.
"""

from __future__ import annotations

import builtins
import re

import numpy as np

from .trajectory_io import Trajectory, expected_num_samples, validate_against_control_params


# 물리→USD 동기화 스위치. rollout 중 sim.step() 이 매 스텝 물리 상태를 USD로 써넣으면
# property 창 / transform 기즈모의 리빌드 코루틴이 스케줄되고, 우리 asyncio task 실행
# 중이라 "Cannot enter into task ... while another task is being executed" 가 폭주해
# 이벤트 루프가 깨진다 (Isaac Lab 이 headless 로 도는 이유). 튜닝 중에는 꺼두고
# 끝나면 되돌린다. 끄면 뷰포트에 손 움직임이 안 보이지만 튜닝엔 상관없다.
_PHYSX_USD_SYNC_KEYS = (
    "/physics/updateToUsd",
    "/physics/updateVelocitiesToUsd",
    "/physics/updateParticlesToUsd",
    "/physics/updateForceSensorsToUsd",
)


def _set_physx_usd_sync(enabled: bool) -> None:
    try:
        import carb

        settings = carb.settings.get_settings()
        for key in _PHYSX_USD_SYNC_KEYS:
            try:
                settings.set(key, bool(enabled))
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


# 대량 prim 삭제(복제 env 스코프)를 실행 중인 asyncio task 안에서 하면 property
# 창 리빌드 코루틴이 폭주한다("Cannot enter into task"). 삭제를 다음 Kit 프레임
# (= task 밖)으로 미룬다. 서브스크립션이 콜백 발화 전에 GC 되지 않도록 모듈
# 리스트로 붙잡아 둔다.
_PENDING_CLEANUPS: list = []


def _defer_call(fn) -> None:
    """fn 을 다음 Kit 프레임(= 실행 중 asyncio task 밖)에서 한 번 호출한다."""
    try:
        import omni.kit.app

        state: dict = {}

        def _cb(_e) -> None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
            state["sub"] = None
            try:
                _PENDING_CLEANUPS.remove(state)
            except ValueError:
                pass

        state["sub"] = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(_cb, name="twin_tuner_deferred")
        )
        _PENDING_CLEANUPS.append(state)
    except Exception:  # noqa: BLE001
        # 미루기 실패하면 즉시 호출로 폴백.
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass

# MuJoCo(궤적 CSV)와 Isaac(트윈 USD) 쪽의 Allegro Hand 관절 이름 규칙이 서로 다름:
#   - MuJoCo:  ff/mf/rf/th + j0..j3 (예: 'ffj0'), 손가락별로 묶여 있음.
#   - Isaac:   index/middle/ring/thumb + joint_0..joint_3 (예: 'index_joint_0'),
#              관절 인덱스별로 묶여 있음.
# 같은 물리적 관절이라도 이름과 컬럼 순서가 다르므로, 문자열이 같은지 비교하는 대신
# 이름을 서로 매핑해서 재배열해야 한다. (조인트 0~3의 base->tip 대응 순서는
# 두 컨벤션에서 동일하다고 가정 — Allegro Hand 표준 조인트 번호.)
_MUJOCO_TO_ISAAC_FINGER = {"ff": "index", "mf": "middle", "rf": "ring", "th": "thumb"}


def _mujoco_name_to_isaac(name: str) -> str:
    finger_code, rest = name[:2], name[2:]
    if finger_code not in _MUJOCO_TO_ISAAC_FINGER or not rest.startswith("j"):
        raise ValueError(f"Unrecognized MuJoCo joint name: {name!r}")
    return f"{_MUJOCO_TO_ISAAC_FINGER[finger_code]}_joint_{rest[1:]}"


# DG-3F 그리퍼는 USD 에셋마다 관절 이름 규칙이 다르다:
#   - delto.usd / delto_t.usd / delto_gripper_3f.usd
#     (URDF 임포트):                          'F1M1' … 'F3M4'    (+ 고정 조인트 TIP1/2/3)
#                                             ← DG-3F CSV 로더가 뽑는 이름
#   - dg3f.usd (isaacsim 패키지):            'j_1_1' … 'j_3_4'
#   - gripper 프리픽스가 붙은 USD 에셋:        'gripper_f1m1_joint' … 'gripper_f3m4_joint'
#     (+ 고정 조인트 gripper_tip1/2/3_joint)
# 셋 다 손가락 N(1-3) × 관절 M(1-4), base→tip 순서도 같으므로 순수 개명이다.
# CSV 로더는 'F{N}M{M}' 로 뽑지만, 아래 파서는 'j_N_M' 형태도 받아들여서
# dg3f.usd 처럼 'j_1_1' 이름을 쓰는 트윈과도 계속 매칭된다.
_DG3F_CANON_RE = re.compile(r"^j_(\d+)_(\d+)$")
_DG3F_FNMM_RE = re.compile(r"^F(\d+)M(\d+)$")


def _dg3f_finger_motor(name: str) -> "tuple[str, str]":
    """DG-3F 관절 이름에서 (손가락N, 관절M) 문자열을 뽑는다.

    받아들이는 형태: 'F1M1'(URDF 임포트 / CSV 로더가 뽑는 이름) 또는
    'j_1_1'(dg3f.usd isaacsim 패키지 정규형).
    """
    m = _DG3F_FNMM_RE.match(name) or _DG3F_CANON_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized DG-3F joint name: {name!r}")
    return m.group(1), m.group(2)


def _dg3f_to_fnmn(name: str) -> str:
    n, m = _dg3f_finger_motor(name)
    return f"F{n}M{m}"


def _dg3f_to_canon(name: str) -> str:
    n, m = _dg3f_finger_motor(name)
    return f"j_{n}_{m}"


def _dg3f_to_gripper_fnmm(name: str) -> str:
    n, m = _dg3f_finger_motor(name)
    return f"gripper_f{n}m{m}_joint"


def _map_csv_names_to_sim(csv_names: list, sim_joint_names: list) -> list:
    """CSV 관절 이름 목록을 트윈 DOF 이름 목록에 1:1 대응시킨다.

    순서대로 시도:
      1) 이름이 이미 트윈 DOF 이름과 그대로 일치 — DG-3F('F1M1' …, CSV 로더
         기본) 나, 이미 Isaac 컨벤션으로 뽑아둔 CSV.
      2) MuJoCo Allegro 컨벤션('ffj0' …)을 Isaac('index_joint_0')으로 변환 후 일치.
      3) DG-3F 이름('F1M1' / 'j_1_1' …)을 URDF 임포트 이름('F1M1' …)으로 변환 후 일치.
      4) DG-3F 이름을 dg3f.usd 정규 이름('j_1_1' …)으로 변환 후 일치.
      5) DG-3F 이름을 gripper 프리픽스 이름('gripper_f1m1_joint' …)으로 변환 후 일치.
    모두 아니면 ValueError.

    반환: csv_names 와 같은 길이·순서의 리스트 — 각 원소가 대응되는 트윈 DOF 이름.
    """
    sim_set = set(sim_joint_names)

    if set(csv_names) == sim_set:
        return list(csv_names)

    try:
        translated = [_mujoco_name_to_isaac(n) for n in csv_names]
    except ValueError:
        translated = None
    if translated is not None and set(translated) == sim_set:
        return translated

    try:
        translated_dg3f = [_dg3f_to_fnmn(n) for n in csv_names]
    except ValueError:
        translated_dg3f = None
    if translated_dg3f is not None and set(translated_dg3f) == sim_set:
        return translated_dg3f

    try:
        translated_canon = [_dg3f_to_canon(n) for n in csv_names]
    except ValueError:
        translated_canon = None
    if translated_canon is not None and set(translated_canon) == sim_set:
        return translated_canon

    try:
        translated_gripper = [_dg3f_to_gripper_fnmm(n) for n in csv_names]
    except ValueError:
        translated_gripper = None
    if translated_gripper is not None and set(translated_gripper) == sim_set:
        return translated_gripper

    raise ValueError(
        "Could not match trajectory CSV (q_ref or q_real) joint names to the "
        "twin Articulation DOF names.\n"
        f"  CSV joint names:        {sorted(csv_names)}\n"
        f"  MuJoCo->Isaac mapping:  "
        f"{sorted(translated) if translated is not None else '(not convertible)'}\n"
        f"  DG-3F->FNMM mapping:    "
        f"{sorted(translated_dg3f) if translated_dg3f is not None else '(not convertible)'}\n"
        f"  DG-3F->j_N_M mapping:   "
        f"{sorted(translated_canon) if translated_canon is not None else '(not convertible)'}\n"
        f"  DG-3F->gripper mapping: "
        f"{sorted(translated_gripper) if translated_gripper is not None else '(not convertible)'}\n"
        f"  twin DOF:               {sorted(sim_joint_names)}\n"
        "Check the CSV loader so column names come out matching the twin joint names."
    )


def _reorder_to_sim_joint_order(traj: Trajectory, sim_joint_names: list) -> np.ndarray:
    """traj.positions 컬럼을 sim_joint_names(트윈 DOF) 순서에 맞게 재배열해서 반환.

    traj 는 q_ref 든 q_real 이든 상관없다 — 둘 다 같은 CSV 로더가 뽑아낸
    Trajectory 이고, 이름 매칭 로직도 동일하게 적용된다.
    """
    mapped_names = _map_csv_names_to_sim(traj.joint_names, sim_joint_names)
    name_to_col = {name: i for i, name in enumerate(mapped_names)}
    reorder_idx = [name_to_col[name] for name in sim_joint_names]
    return traj.positions[:, reorder_idx]


async def run_trajectory_on_twin(
    articulation_prim_path: str,
    q_ref: Trajectory,
    control_hz: float = 333.0,
    measurement_length_s: float = 3.0,
    show_in_viewport: bool = False,
) -> Trajectory:
    """
    q_ref 를 매 스텝 관절 위치 target으로 트윈 Articulation에 주입하고,
    그 결과 트윈의 실제 관절 각도(q_sim)를 기록해서 반환한다.

    (q_ref 는 "twin에 명령으로 넣는 궤적" — 실측 정답값 q_real 과는 다른 개념이다.
    q_real 과의 비교/loss 계산은 이 함수 밖에서, 별도로 로드한 q_real 을 가지고 한다.)

    전제:
      - Isaac Sim 스테이지가 이미 열려 있고, articulation_prim_path 위치에
        트윈 Articulation(USD)이 이미 존재함.
      - q_ref 의 관절 이름이 MuJoCo Allegro 컨벤션(ff/mf/rf/th + j0..j3)임.
        컬럼 순서는 몰라도 됨 — 이름을 Isaac 컨벤션으로 변환해서 트윈 DOF
        순서에 맞게 자동으로 재배열한다 (매칭이 안 되면 에러로 막힘).

    show_in_viewport: True 면 재생 중 물리→USD 동기화를 켜서 뷰포트에 손 움직임이
      보인다. OmniGraph/ROS2 없는 URDF 임포트 에셋에서만 안전 (기본 False).

    반환: q_sim (Trajectory) - 트윈이 실제로 도달한 관절 각도 시계열.

    이 함수가 async 인 이유 + sim.step()을 항상 render=False로 부르는 이유:
    이 함수는 UI 버튼 클릭 콜백에서(asyncio task로) 호출되는데, Kit(GUI
    Extension 런타임)은 자기 메인 루프에서 이미 app.update()를 계속 돌리고
    있다. sim.step(render=True)는 내부적으로 app.update()를 다시 호출하는데,
    이걸 Kit이 이미 처리 중인 프레임 도중(우리 코루틴이든 동기 콜백이든)
    호출하면 이벤트 루프/렌더 루프를 재귀적으로(nested) 재진입하는 셈이 된다.
    이게 "cmdBegin을 cmdEnd 전에 또 호출함" 렌더러 커맨드 리스트 손상
    -> Vulkan 크래시, 그리고 asyncio의 "Cannot enter into task ... while
    another task is being executed" 에러 둘 다의 원인이었다 (standalone
    스크립트의 `while simulation_app.is_running(): world.step(render=True)`
    패턴은 그 루프 자체가 최상위 프레임 펌프라 문제없지만, GUI Extension
    안에서는 Kit이 이미 그 역할을 하고 있어서 절대 안 되는 패턴).
    그래서 물리는 render=False로만 진행시키고(자체적으로 app.update()를
    부르지 않음), 화면 갱신은 매 스텝 뒤 `await app.next_update_async()`로
    Kit 자신의 다음 프레임을 기다리는 것으로 대신한다 - 그 프레임에서
    Kit이 어차피 현재 스테이지 상태를 그리므로 시각적으로는 매 스텝
    손가락이 움직이는 게 그대로 보인다.
    """
    validate_against_control_params(q_ref, control_hz, measurement_length_s)

    # Isaac Sim 런타임 안에서만 import 가능하므로 함수 내부에서 import.
    import omni.kit.app
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    app = omni.kit.app.get_app()

    dt = 1.0 / control_hz
    num_steps = expected_num_samples(control_hz, measurement_length_s)

    # SimulationContext.__init__ 은 builtins.ISAAC_LAUNCHED_FROM_TERMINAL 이
    # True(기본값)일 때 내부 _init_stage()를 건너뛴다. _init_stage()는 단순히
    # physics_dt만 세팅하는 게 아니라, SingleArticulation.initialize()가 의존하는
    # physx tensor API의 "simulation view" 구독까지 같이 해준다.
    # 이걸 우리가 PhysicsContext(...)를 직접 만들어 흉내 냈더니 physics_dt는
    # 맞아도 simulation view 쪽 초기화가 빠져서 'Simulation view object is
    # invalidated' -> 렌더러 커맨드 리스트 손상 -> Vulkan 크래시로 이어졌다.
    # 대신 이 플래그 자체를 False로 만들어서 Isaac Sim 자신의 검증된
    # _init_stage() 로직을 그대로 타게 한다.
    builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False

    sim = SimulationContext.instance()
    if sim is not None and sim.get_physics_context() is None:
        # 이전 실행에서 반쪽짜리로 초기화된 좀비 인스턴스일 수 있으니 버리고 새로 만든다.
        SimulationContext.clear_instance()
        sim = None

    if sim is None:
        sim = SimulationContext(physics_dt=dt, rendering_dt=dt)
    else:
        sim.set_simulation_dt(physics_dt=dt, rendering_dt=dt)

    # 중요: initialize()는 물리 시뮬레이션이 play 상태여야 동작한다.
    # (physics simulation view가 play 시점에 생성되고, 그걸 articulation view가
    #  참조하기 때문 - 순서를 바꾸면 'NoneType' object has no attribute
    #  'create_articulation_view' 에러가 난다.)
    # play() 이후에 나는 모든 예외(관절 이름 불일치 포함)에도 sim.stop()이
    # 반드시 실행되도록, play() 직후부터 try/finally로 감싼다.
    # 물리→USD 동기화 끔: 매 sim.step() 의 USD writeback 이 property/stage/viewport
    # 위젯 리빌드 코루틴을 스케줄하고, 그게 이 asyncio task 실행 중에 재진입하면
    # "Cannot enter into task ... while another task is being executed" 폭주로
    # 이벤트 루프가 깨진다 (OmniGraph/ROS2 가 붙은 에셋 — 예: DG-3F dg3f.usd —
    # 에서 특히 심함). 수집 중엔 뷰포트에 손 움직임이 안 보여도 무방하다.
    #
    # show_in_viewport=True 면 동기화를 켠 채로 재생해서 매 스텝 손 움직임이
    # 뷰포트에 그대로 보인다 (render=False + await next_update_async() 뒤 Kit 이
    # 갱신된 스테이지를 그린다). OmniGraph 없는 URDF 임포트 에셋
    # (delto.usd / delto_gripper_3f.usd 등) 에서만 안전 — dg3f.usd 는 위 폭주 위험.
    _set_physx_usd_sync(bool(show_in_viewport))
    sim.play()
    try:
        twin = SingleArticulation(prim_path=articulation_prim_path)
        twin.initialize()

        sim_joint_names = list(twin.dof_names)
        q_ref_positions = _reorder_to_sim_joint_order(q_ref, sim_joint_names)

        num_joints = len(sim_joint_names)
        q_sim_positions = np.zeros((num_steps, num_joints), dtype=np.float64)
        q_sim_timestamps = np.arange(num_steps, dtype=np.float64) * dt

        for i in range(num_steps):
            target = q_ref_positions[i]
            twin.apply_action(ArticulationAction(joint_positions=target))
            # render=False: app.update() 재진입(렌더러 손상/asyncio 에러의 원인)을
            # 피하기 위해 물리만 진행시킨다. 화면 갱신은 아래 next_update_async()가
            # 기다리는 Kit의 다음 프레임에서 자연히 이루어진다.
            sim.step(render=False)
            q_sim_positions[i] = twin.get_joint_positions()
            # 매 스텝마다 Kit의 다음 프레임까지 양보한다. 이렇게 안 하면
            # 999 스텝이 이 코루틴 안에서 그대로 블로킹으로 다 돌아버려서
            # UI가 멈추고 화면도 끝날 때까지 갱신되지 않는다.
            await app.next_update_async()
    finally:
        sim.stop()
        _set_physx_usd_sync(True)  # 물리→USD 동기화 원복 (뷰포트 갱신 복구)

    return Trajectory(
        joint_names=sim_joint_names,
        timestamps=q_sim_timestamps,
        positions=q_sim_positions,
    )


# ======================================================================
# Step 2 - CMA-ES 튜닝용 rollout (게인만 바꿔 같은 궤적을 반복 재생)
# ======================================================================


class TwinRolloutRunner:
    """트윈 Articulation을 한 번 play/initialize 해두고, 게인만 바꿔가며
    q_ref 궤적(명령값)을 여러 번(수천 번) 재생하는 러너.

    (q_real — loss 비교용 실측 정답 궤적 — 은 이 러너가 모른다. 호출자가
    rollout()이 반환한 q_sim을 q_real과 직접 비교한다.)

    Step 1의 `run_trajectory_on_twin` 은 rollout 1회 + 매번 sim 초기화지만,
    CMA-ES는 rollout을 population*generations 번 돌리므로:
      - sim/articulation 초기화는 setup()에서 딱 1번만 한다.
      - rollout()마다 (1) 관절 상태를 t=0 값으로 리셋 → 결정론적 목적함수,
        (2) stiffness/damping 세팅, (3) 궤적 재생, (4) q_sim 반환.
      - 매 스텝 await 하던 걸 yield_every 스텝마다로 줄여 속도를 확보한다
        (물리는 계속 render=False라 app.update() 재진입 크래시는 없음. 원래
        매 스텝 await는 화면 갱신용이었을 뿐).

    이 클래스는 단일 env(env_count == 1). 여러 env를 physX cloner로 병렬 재생하려면
    ParallelTwinRolloutRunner 를 쓴다.
    """

    def __init__(
        self,
        articulation_prim_path: str,
        q_ref: Trajectory,
        control_hz: float = 333.0,
        measurement_length_s: float = 3.0,
    ) -> None:
        validate_against_control_params(q_ref, control_hz, measurement_length_s)
        self._prim_path = articulation_prim_path
        self._q_ref = q_ref
        self._control_hz = control_hz
        self._measurement_length_s = measurement_length_s
        self._dt = 1.0 / control_hz
        self._num_steps = expected_num_samples(control_hz, measurement_length_s)

        self._app = None
        self._sim = None
        self._twin = None
        self._controller = None
        self.sim_joint_names: list = []
        self.q_ref_aligned: "np.ndarray | None" = None  # (num_steps, num_joints)
        self.default_stiffness: "np.ndarray | None" = None
        self.default_damping: "np.ndarray | None" = None
        self._q0: "np.ndarray | None" = None
        self._qd0: "np.ndarray | None" = None

    def build_scene(self) -> None:
        """단일 env는 스테이지 복제가 없으므로 no-op (ParallelTwinRolloutRunner
        와 인터페이스를 맞추기 위한 자리)."""
        return

    async def setup(self) -> None:
        import omni.kit.app
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation

        self._app = omni.kit.app.get_app()

        # run_trajectory_on_twin 과 동일한 초기화 시퀀스 (주석은 그쪽 참고).
        builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False

        sim = SimulationContext.instance()
        if sim is not None and sim.get_physics_context() is None:
            SimulationContext.clear_instance()
            sim = None
        if sim is None:
            sim = SimulationContext(physics_dt=self._dt, rendering_dt=self._dt)
        else:
            sim.set_simulation_dt(physics_dt=self._dt, rendering_dt=self._dt)
        self._sim = sim

        # 물리→USD 동기화 끔. 예전엔 단일 env 는 켜둬도 안 터졌지만, OmniGraph/ROS2
        # 가 붙은 에셋(예: DG-3F dg3f.usd)에서는 단일 env 라도 매 step USD writeback 이
        # property/stage 위젯 리빌드 코루틴을 스케줄해 "Cannot enter into task" 폭주로
        # 이어진다. 튜닝 중 뷰포트에 손 움직임이 안 보여도 무방 — q_sim 만 있으면 된다.
        _set_physx_usd_sync(False)

        sim.play()
        # play 직후 물리 sim view가 잡히도록 Kit 프레임 한 번 양보 후 initialize.
        await self._app.next_update_async()

        self._twin = SingleArticulation(prim_path=self._prim_path)
        self._twin.initialize()

        self.sim_joint_names = list(self._twin.dof_names)
        self.q_ref_aligned = _reorder_to_sim_joint_order(
            self._q_ref, self.sim_joint_names
        )

        # 리셋 기준이 될 t=0 상태를 저장 (매 rollout 시작 시 여기로 되돌림).
        self._q0 = np.asarray(self._twin.get_joint_positions(), dtype=np.float64).copy()
        qd0 = self._twin.get_joint_velocities()
        self._qd0 = (
            np.zeros(len(self.sim_joint_names), dtype=np.float64)
            if qd0 is None
            else np.asarray(qd0, dtype=np.float64).copy()
        )

        # 현재(튜닝 전) 게인 = CMA-ES 시작점. Isaac Sim ArticulationController API.
        self._controller = self._twin.get_articulation_controller()
        try:
            kps, kds = self._controller.get_gains()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not read the twin's current drive gains "
                "(ArticulationController.get_gains). Check the Isaac Sim "
                f"version's API. Cause: {e}"
            ) from e
        self.default_stiffness = np.asarray(kps, dtype=np.float64).ravel().copy()
        self.default_damping = np.asarray(kds, dtype=np.float64).ravel().copy()

    async def rollout(
        self, stiffness: "np.ndarray", damping: "np.ndarray", yield_every: int = 50
    ) -> "np.ndarray":
        """게인 1세트로 q_ref 를 재생하고 도달 관절각 시계열 (num_steps, J) 반환.

        yield_every: 이 스텝 수마다 Kit 프레임에 양보(화면 갱신 + task 취소 반응).
        작을수록 움직임이 부드럽게 보이지만 rollout 이 느려진다. 단일 env 기본 50.

        주의: 이 값이 num_steps 보다 크면(짧은 궤적) 단 한 번도 양보를 안 해서,
        이 rollout() 호출 하나가 통째로 Kit UI 스레드를 막아버린다 — CMA-ES가
        세대마다(그리고 세대 안에서 후보마다) 이걸 반복 호출하므로, 최악의 경우
        튜닝 전체가 한 프레임도 안 그려진 채로 끝날 때까지 안 돌아와서 OS가
        "응답 없음"으로 판단해 창이 멈춘 것처럼 보인다. 그래서 실제로 쓰는 값은
        궤적 길이에 상관없이 최소 4~5번은 양보하도록 아래에서 클램프한다.
        """
        from isaacsim.core.utils.types import ArticulationAction

        if self._twin is None:
            raise RuntimeError("setup() must be called first.")

        stiffness = np.asarray(stiffness, dtype=np.float64).ravel()
        damping = np.asarray(damping, dtype=np.float64).ravel()

        # (1) t=0 상태로 리셋 — rollout끼리 독립적이고 결정론적이도록.
        self._twin.set_joint_positions(self._q0)
        self._twin.set_joint_velocities(self._qd0)

        # (2) 이번 후보 게인 적용.
        self._controller.set_gains(kps=stiffness, kds=damping)

        num_joints = len(self.sim_joint_names)
        q_sim = np.zeros((self._num_steps, num_joints), dtype=np.float64)

        effective_yield_every = (
            max(1, min(yield_every, self._num_steps // 4)) if yield_every > 0 else 0
        )

        # (3) 궤적 재생.
        for i in range(self._num_steps):
            self._twin.apply_action(
                ArticulationAction(joint_positions=self.q_ref_aligned[i])
            )
            self._sim.step(render=False)
            q_sim[i] = self._twin.get_joint_positions()
            # 매 스텝이 아니라 가끔만 Kit에 양보 (UI 반응 + task 취소 가능하도록).
            if (
                effective_yield_every > 0
                and (i % effective_yield_every) == (effective_yield_every - 1)
            ):
                await self._app.next_update_async()

        return q_sim

    def teardown(self) -> None:
        _set_physx_usd_sync(True)  # 물리→USD 동기화 원복 (뷰포트 갱신 복구)
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:  # noqa: BLE001
                pass
        self._sim = None
        self._twin = None
        self._controller = None


async def author_drive_gains_to_usd(
    articulation_prim_path: str,
    stiffness: "np.ndarray",
    damping: "np.ndarray",
    control_hz: float = 333.0,
) -> None:
    """튜닝으로 찾은 게인을 트윈에 실제 적용 + USD에 저장 ("Apply to twin").

    play/initialize 후 ArticulationController.set_gains(save_to_usd=True) 로
    적용한다. save_to_usd=True 라 단위 변환(rad<->deg 등)은 Isaac Sim이 알아서
    처리하고, stop() 후에도 스테이지에 값이 남는다.
    """
    import omni.kit.app
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation

    app = omni.kit.app.get_app()
    dt = 1.0 / control_hz

    builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False
    sim = SimulationContext.instance()
    if sim is not None and sim.get_physics_context() is None:
        SimulationContext.clear_instance()
        sim = None
    if sim is None:
        sim = SimulationContext(physics_dt=dt, rendering_dt=dt)
    else:
        sim.set_simulation_dt(physics_dt=dt, rendering_dt=dt)

    sim.play()
    try:
        await app.next_update_async()
        twin = SingleArticulation(prim_path=articulation_prim_path)
        twin.initialize()
        controller = twin.get_articulation_controller()
        controller.set_gains(
            kps=np.asarray(stiffness, dtype=np.float64).ravel(),
            kds=np.asarray(damping, dtype=np.float64).ravel(),
            save_to_usd=True,
        )
    finally:
        sim.stop()


# ======================================================================
# Step 2 - 병렬 env rollout (physX cloner)
# ======================================================================


class ParallelTwinRolloutRunner:
    """트윈 손을 env_count 개 복제해 한 PhysX 씬에서 동시에 재생하는 러너.

    한 세대(population 개 후보)를 배치 rollout 1~몇 회로 평가한다. 후보 수가
    env_count 보다 많으면 호출자가 env_count 크기로 잘라서 rollout_batch 를
    여러 번 부른다.

    옵션 A(안정 우선): define_base_env → 템플릿 env_0(CopyPrim) →
    GridCloner.clone(replicate_physics=False, copy_from_source=True) →
    관계 기반 CollisionGroup 자기필터 → 배치 Articulation 뷰.
    (replicate_physics=True / cloner.filter_collisions 는 이 환경에서 USD
    path-node 재귀 소멸 스택오버플로로 하드 크래시 이력 → 쓰지 않음.)

    동작:
      build_scene()  ← 동기. asyncio task 시작 전에 UI 콜백에서 호출
                       (스테이지 복제 알림 폭주 ↔ 실행 중 task 충돌 회피).
        - 사용자 손을 _twin_tuner_envs/env_0/<leaf> 로 1벌 복사 → env_1..N-1
          독립 사본 복제 + 격자 배치 + cross-env 충돌 격리.
          사용자 원본은 안 건드림(뷰에도 미포함).
      setup()        ← async.
        - enable_gpu_dynamics(무시가능) → SimulationContext play → 배치 뷰 initialize
        - (고정베이스면) set_world_poses 로 격자 재분리
        - env 0 기준 t=0 상태·기본 게인 저장.
      rollout_batch(stiffness_batch (P,J), damping_batch, ...)
        - 앞의 P개 env에 후보 게인, 나머지 env는 기본 게인(놀림).
        - 전체 env를 t=0으로 리셋 → q_ref 를 모든 env에 같은 target으로 재생.
        - 반환 q_sim: (P, num_steps, J).
      teardown()
        - sim.stop() 후 스크래치 스코프 삭제.

    주의: 배치 Articulation 뷰 API는 Isaac Sim 버전에 민감하다. 5.x
    (isaacsim.core.*) 기준으로 작성했고, 호출이 실패하면 임의로 다른 이름을
    지어내지 말고 그 버전 문서를 확인해야 한다.
    """

    ENV_SCOPE = "/World/_twin_tuner_envs"

    def __init__(
        self,
        articulation_prim_path: str,
        q_ref: Trajectory,
        control_hz: float = 333.0,
        measurement_length_s: float = 3.0,
        env_count: int = 128,
        env_spacing: float = 1.0,
    ) -> None:
        validate_against_control_params(q_ref, control_hz, measurement_length_s)
        if env_count < 2:
            raise ValueError("ParallelTwinRolloutRunner requires env_count >= 2.")
        self._src_path = articulation_prim_path
        self._hand_leaf = articulation_prim_path.rstrip("/").rsplit("/", 1)[-1]
        self._q_ref = q_ref
        self._control_hz = control_hz
        self._measurement_length_s = measurement_length_s
        self._env_count = int(env_count)
        self._env_spacing = float(env_spacing)
        self._dt = 1.0 / control_hz
        self._num_steps = expected_num_samples(control_hz, measurement_length_s)

        self._app = None
        self._sim = None
        self._view = None
        self._delete_prim = None
        self._env_positions = None
        self._scene_built = False
        self.sim_joint_names: list = []
        self.q_ref_aligned: "np.ndarray | None" = None
        self.default_stiffness: "np.ndarray | None" = None
        self.default_damping: "np.ndarray | None" = None
        self._q0: "np.ndarray | None" = None
        self._qd0: "np.ndarray | None" = None

    @property
    def env_count(self) -> int:
        return self._env_count

    def _resolve_source_asset(self, stage):
        """사용자 손 프림이 참조하는 소스 USD 파일 경로 + 파일 내부 프림 경로를 찾는다.

        (self._src_path 또는 그 조상에 걸린 reference/payload arc 를 훑는다.)
        찾으면 (asset_identifier, internal_prim_path). 못 찾으면 RuntimeError.
        """
        from pxr import Pcp, Sdf, Usd

        node = stage.GetPrimAtPath(self._src_path)
        while node and node.IsValid() and node.GetPath() != Sdf.Path.absoluteRootPath:
            try:
                arcs = Usd.PrimCompositionQuery(node).GetCompositionArcs()
            except Exception:  # noqa: BLE001
                arcs = []
            for arc in arcs:
                if arc.GetArcType() not in (Pcp.ArcTypeReference, Pcp.ArcTypePayload):
                    continue
                layer = arc.GetTargetLayer()
                ident = getattr(layer, "realPath", "") or getattr(layer, "identifier", "")
                if not ident:
                    continue
                target_prim = str(arc.GetTargetPrimPath())
                # arc 가 조상에 걸려 있으면, src 까지의 나머지 경로를 뒤에 붙인다.
                rel = self._src_path[len(str(node.GetPath())):]
                internal = (target_prim.rstrip("/") + rel) if rel else target_prim
                return ident, internal
            node = node.GetParent()

        raise RuntimeError(
            f"Could not find a source USD file (reference/payload) from '{self._src_path}'. "
            "If this hand is not a file reference it cannot be cloned in parallel (GPU) mode "
            "- run with Env=1 (single), or save the hand as a .usd file and reference it."
        )

    def build_scene(self) -> None:
        """스테이지에 env_count 벌 복제본을 만든다 (동기 — asyncio task 시작 전에
        UI 콜백에서 직접 호출; 스테이지 변경 알림 폭주 ↔ 실행 중 task 충돌 회피).

        옵션 A (안정 우선):
          - 템플릿 env_0 에 사용자 손을 CopyPrim(exclusive_select=False)로 1벌 복사
          - GridCloner.clone(replicate_physics=False, copy_from_source=True) 로 나머지
            복제 + 격자 배치. replicate_physics=True 는 이 환경(5.1-rc + payload 손)에서
            USD Sdf path-node 재귀 소멸 스택오버플로로 하드 크래시 이력 → 쓰지 않음.
          - 충돌 격리는 cloner.filter_collisions (path-expression → 크래시 이력) 대신
            관계(relationship) 기반 CollisionGroup 하나를 자기 자신에 필터링해서
            "env 손끼리는 서로 충돌 안 함(각자 self-collision/ground 는 유지)".
        """
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
        from pxr import Usd, UsdGeom, UsdPhysics

        self._delete_prim = delete_prim
        stage = omni.usd.get_context().get_stage()

        if not is_prim_path_valid(self._src_path):
            raise RuntimeError(
                f"Source prim '{self._src_path}' not found in the stage "
                "(check the Robot Selection value)."
            )

        if is_prim_path_valid(self.ENV_SCOPE):
            delete_prim(self.ENV_SCOPE)

        env_ns = self.ENV_SCOPE
        template_env = f"{env_ns}/env_0"
        template_hand = f"{template_env}/{self._hand_leaf}"
        env_paths = [f"{env_ns}/env_{i}" for i in range(self._env_count)]

        cloner = GridCloner(spacing=self._env_spacing)
        cloner.define_base_env(env_ns)
        UsdGeom.Xform.Define(stage, template_env)

        ok, _ = omni.kit.commands.execute(
            "CopyPrim",
            path_from=self._src_path,
            path_to=template_hand,
            exclusive_select=False,
        )
        if not ok or not is_prim_path_valid(template_hand):
            raise RuntimeError(f"CopyPrim failed: '{self._src_path}' -> '{template_hand}'")

        # replicate_physics=False → root_path 불필요. copy_from_source=True → 각 env가
        # 독립 사본(instanceable+payload 조합 회피).
        self._env_positions = cloner.clone(
            source_prim_path=template_env,
            prim_paths=env_paths,
            replicate_physics=False,
            copy_from_source=True,
        )

        made = [p for p in env_paths if is_prim_path_valid(p)]
        if len(made) != self._env_count:
            raise RuntimeError(
                f"GridCloner created only {len(made)}/{self._env_count} env prims."
            )

        # 충돌 격리 (관계 기반, path-expression 안 씀).
        try:
            grp_path = f"{env_ns}/noCrossEnvCollision"
            grp = UsdPhysics.CollisionGroup.Define(stage, grp_path)
            try:
                coll = grp.GetCollidersCollectionAPI()
            except Exception:  # noqa: BLE001
                coll = Usd.CollectionAPI.Apply(grp.GetPrim(), "colliders")
            includes = coll.CreateIncludesRel()
            for ep in env_paths:
                includes.AddTarget(ep)
            grp.CreateFilteredGroupsRel().AddTarget(grp_path)
        except Exception as e:  # noqa: BLE001
            import carb

            carb.log_warn(
                f"[Parameter Tuning] failed to create cross-env collision groups (depends on grid spacing): {e}"
            )

        self._scene_built = True

    async def setup(self) -> None:
        import carb
        import omni.kit.app
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation

        if not getattr(self, "_scene_built", False):
            raise RuntimeError("build_scene() must be called first.")

        self._app = omni.kit.app.get_app()
        builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False
        env_ns = self.ENV_SCOPE

        sim = SimulationContext.instance()
        if sim is not None and sim.get_physics_context() is None:
            SimulationContext.clear_instance()
            sim = None
        if sim is None:
            sim = SimulationContext(physics_dt=self._dt, rendering_dt=self._dt)
        else:
            sim.set_simulation_dt(physics_dt=self._dt, rendering_dt=self._dt)
        self._sim = sim

        # 튜닝 중 물리→USD 동기화 끔: 안 그러면 배치 뷰의 상태 write + step 마다
        # property/gizmo 리빌드 코루틴이 스케줄돼 "Cannot enter into task" 폭주로
        # 이벤트 루프가 깨진다 (Isaac Lab headless 와 동일 취지).
        _set_physx_usd_sync(False)

        try:
            sim.get_physics_context().enable_gpu_dynamics(True)
        except Exception as e:  # noqa: BLE001
            carb.log_warn(f"[Parameter Tuning] enable_gpu_dynamics failed (ignored): {e}")

        sim.play()
        await self._app.next_update_async()

        self._view_expr = f"{env_ns}/env_[0-9]+/{self._hand_leaf}"
        self._view = Articulation(
            prim_paths_expr=self._view_expr,
            name="twin_tuner_parallel_view",
        )
        self._view.initialize()

        n_found = int(getattr(self._view, "count", self._env_count))
        if n_found != self._env_count:
            raise RuntimeError(
                f"Cloned {self._env_count} envs but the batch view (expr='{self._view_expr}') "
                f"matched only {n_found}. Check that ArticulationRootAPI is kept on the clones."
            )

        # 고정베이스 손이면 격자 오프셋이 무시돼 겹칠 수 있다 → 뷰로 다시 벌린다.
        try:
            if self._env_positions is not None:
                pos = np.asarray(self._env_positions, dtype=np.float64)
                if pos.shape == (self._env_count, 3):
                    self._view.set_world_poses(positions=pos)
                    self._sim.step(render=False)
        except Exception as e:  # noqa: BLE001
            carb.log_warn(
                f"[Parameter Tuning] failed to separate envs via set_world_poses (ignored): {e}"
            )

        self.sim_joint_names = list(self._view.dof_names)
        self.q_ref_aligned = _reorder_to_sim_joint_order(
            self._q_ref, self.sim_joint_names
        )

        q0 = np.asarray(self._view.get_joint_positions(), dtype=np.float64)  # (N, J)
        self._q0 = q0[0].copy()
        self._qd0 = np.zeros(len(self.sim_joint_names), dtype=np.float64)

        try:
            kps, kds = self._view.get_gains()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not read the clone view's current drive gains (Articulation.get_gains). "
                f"Check the Isaac Sim version's API. Cause: {e}"
            ) from e
        kps = np.asarray(kps, dtype=np.float64)
        kds = np.asarray(kds, dtype=np.float64)
        self.default_stiffness = (kps[0] if kps.ndim == 2 else kps).ravel().copy()
        self.default_damping = (kds[0] if kds.ndim == 2 else kds).ravel().copy()

    async def rollout_batch(
        self,
        stiffness_batch: "np.ndarray",
        damping_batch: "np.ndarray",
        yield_every: int = 200,
    ) -> "np.ndarray":
        """P개(<= env_count) 후보 게인으로 동시에 q_ref 재생.

        stiffness_batch / damping_batch: shape (P, J).
        반환: q_sim (P, num_steps, J).

        주의: yield_every(기본 200)가 num_steps 보다 크면 이 호출 안에서
        `await self._app.next_update_async()`가 단 한 번도 안 불린다 — 즉 세대
        하나(=이 호출 1번)가 통째로 Kit UI 스레드를 막는다. 짧은 궤적(예:
        333Hz*3s=999 가 아니라 85 스텝짜리)에서 CMA-ES가 세대를 수십~수백 번
        돌리면, 그 세대들이 전부 한 프레임도 안 그려진 채 이어져서 OS가
        "응답 없음(Wait/Force Quit)" 창을 띄우는 원인이 된다. 궤적 길이에
        상관없이 최소 4~5번은 양보하도록 실제 사용값을 클램프한다.
        """
        if self._view is None:
            raise RuntimeError("setup() must be called first.")

        stiffness_batch = np.atleast_2d(np.asarray(stiffness_batch, dtype=np.float64))
        damping_batch = np.atleast_2d(np.asarray(damping_batch, dtype=np.float64))
        p = stiffness_batch.shape[0]
        n = self._env_count
        j = len(self.sim_joint_names)
        if p > n:
            raise ValueError(f"{p} candidates > env_count {n}. The caller must chunk the batch.")

        # 앞 P개 env = 후보, 나머지 = 기본 게인(놀리는 env).
        kps = np.tile(self.default_stiffness, (n, 1))
        kds = np.tile(self.default_damping, (n, 1))
        kps[:p] = stiffness_batch
        kds[:p] = damping_batch
        self._view.set_gains(kps=kps, kds=kds)

        # 전체 env t=0 리셋.
        self._view.set_joint_positions(np.tile(self._q0, (n, 1)))
        self._view.set_joint_velocities(np.tile(self._qd0, (n, 1)))

        q_sim = np.zeros((p, self._num_steps, j), dtype=np.float64)
        effective_yield_every = (
            max(1, min(yield_every, self._num_steps // 4)) if yield_every > 0 else 0
        )
        for t in range(self._num_steps):
            target = np.tile(self.q_ref_aligned[t], (n, 1))  # 모든 env 같은 command
            self._view.set_joint_position_targets(target)
            self._sim.step(render=False)
            pos = np.asarray(self._view.get_joint_positions(), dtype=np.float64)  # (N, J)
            q_sim[:, t, :] = pos[:p]
            if (
                effective_yield_every > 0
                and (t % effective_yield_every) == (effective_yield_every - 1)
            ):
                await self._app.next_update_async()

        return q_sim

    def teardown(self) -> None:
        # 이 메서드는 extension 에서 _defer_call 로 감싸 asyncio task 밖에서 호출된다.
        # sim.stop() / 스코프 삭제 모두 대량 USD 변경이라 task 안에서 하면 property
        # 창 리빌드 코루틴이 폭주한다.
        if self._sim is not None:
            try:
                self._sim.stop()  # updateToUsd=False 상태에서 stop → writeback 억제
            except Exception:  # noqa: BLE001
                pass
        self._sim = None
        self._view = None
        if self._delete_prim is not None:
            try:
                self._delete_prim(self.ENV_SCOPE)
            except Exception:  # noqa: BLE001
                pass
        _set_physx_usd_sync(True)  # 마지막에 물리→USD 동기화 원복 (뷰포트 갱신 복구)
