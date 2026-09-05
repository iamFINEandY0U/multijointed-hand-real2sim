"""
Twin Param Tuner - Extension 진입점.

데이터 흐름 (q_ref / q_real / q_sim 세 가지를 구분한다):
  - q_ref  : Step 1에서 로드해 트윈에 "명령"으로 주입하는 궤적 CSV.
  - q_sim  : q_ref 를 트윈에 재생했을 때 트윈이 실제로 도달한 관절각(rollout 결과).
  - q_real : 실제 로봇에서 측정된 궤적 CSV. Step 2에서 별도로 로드하며, q_sim과
             비교해 loss를 계산하는 "정답값"으로만 쓰인다(트윈에 명령으로 넣지 않음).

Step 1(데이터 확보, Mode B)은 기능까지 구현 완료 — q_ref CSV 로드 → 트윈 재생 → q_sim 수집.
Step 1에서 "real hardware" 옵션을 켜면, 같은 q_ref/제어주기/측정길이로 실제 Delto
DG-3F 손도 별도 프로세스(ReplayRefTrajectory.py, DG-3F 전용 conda env)로 같이
재생해서 q_real을 수집한다 — Data collection 버튼 한 번으로 q_sim과 q_real이
동시에 채워진다(q_real은 Step 2 'Reference (q_real)' 필드에 자동으로 채워짐).
Step 2(튜닝)는 CMA-ES(vendor된 pycma)로 구현함 — Step 2에서 로드한 q_real 을
기준으로, 관절별 (stiffness, damping) 값을 반복 rollout(q_ref 재생)·loss(q_sim vs
q_real) 계산·분포 갱신으로 정합한다. UI 값이 모두 반영됨: Algorithm(CMA-ES 고정),
Population, Generations, Loss indicator(MSE/RMSE/MAE), Learning parameters(앞 L개
관절만 튜닝, 나머지는 기본 게인 고정), Env(1이면 단일 env, >1이면 physX cloner로
env개 복제해 배치 rollout — 후보 수 > env면 env 크기로 잘라 여러 배치).
Step 3(검증)은 UI + "Matching results" 기능까지 구현함 — Step 2에서 로드한
q_real과 q_sim(트윈 실측)을 비교해서 match score/오차 통계/관절별 오차 표/그래프를
보여줌. "Optimization parameters" 표와 "Apply to twin"은 Step 2 튜닝이 끝나면
best stiffness/damping 으로 채워지고, "Apply to twin"이 그 값을 트윈 drive에
적용(USD 저장)한 뒤 곧바로 원래 q_ref를 새 게인으로 다시 재생해 q_sim_a(적용 후
궤적)를 만들고 self._last_q_sim을 그걸로 갱신한다 — 그래서 이어서 "Matching
results"를 누르면 재수집 없이 바로 "실제 vs 적용 후"가 현재 값으로, "실제 vs
적용 전"이 baseline("was ...")으로 비교된다.
"""

import asyncio
import csv
import math
import os
import shlex
import subprocess
import threading
import time

import carb
import numpy as np
import omni.ext
import omni.ui as ui
import omni.usd
from omni.kit.window.filepicker import FilePickerDialog
from pxr import UsdPhysics

from .cma_tuning import LOSS_KINDS, TuningConfig, compute_loss, run_cma_tuning
from .trajectory_io import (
    describe_csv_format,
    detect_csv_kind,
    expected_num_samples,
    load_hand_trajectory_csv,
    save_trajectory_csv,
)
from .twin_playback import (
    ParallelTwinRolloutRunner,
    TwinRolloutRunner,
    _defer_call,
    _reorder_to_sim_joint_order,
    author_drive_gains_to_usd,
    run_trajectory_on_twin,
)


class _RobotComboModel(ui.AbstractItemModel):
    """스테이지에서 찾은 Articulation Root prim 경로들을 보여주는 드롭다운 모델.

    스테이지에 ArticulationRootAPI를 가진 prim이 하나도 없으면(예: 아무것도 배치하지
    않은 빈 월드) 실제로 존재하지 않는 경로를 그럴듯하게 보여주지 않고, 선택 불가능한
    안내 문구 한 줄만 표시한다. get_selected_path()는 이 상태에서 빈 문자열을 반환해서
    "Error: select a robot" 검증이 정상적으로 걸리게 한다.
    """

    _EMPTY_PLACEHOLDER = "(No robot found in stage)"

    class _PathItem(ui.AbstractItem):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.model = ui.SimpleStringModel(path)

    def __init__(self) -> None:
        super().__init__()
        self._items: list = []
        self._has_valid_paths = False
        self._current_index = ui.SimpleIntModel(0)
        self._current_index.add_value_changed_fn(lambda _m: self._item_changed(None))

    def replace_item_list(self, prim_paths: list) -> None:
        if prim_paths:
            self._items = [self._PathItem(p) for p in prim_paths]
            self._has_valid_paths = True
        else:
            self._items = [self._PathItem(self._EMPTY_PLACEHOLDER)]
            self._has_valid_paths = False
        self._current_index.set_value(0)
        self._item_changed(None)

    def get_selected_path(self) -> str:
        if not self._has_valid_paths:
            return ""
        index = self._current_index.get_value_as_int()
        if 0 <= index < len(self._items):
            return self._items[index].model.get_value_as_string()
        return ""

    def get_item_children(self, item=None):
        return self._items

    def get_item_value_model(self, item=None, column_id: int = 0):
        if item is None:
            return self._current_index
        return item.model


class _StringListComboModel(ui.AbstractItemModel):
    """고정된 문자열 목록(Algorithm, Loss indicator 등)을 보여주는 범용 콤보박스 모델."""

    class _StringItem(ui.AbstractItem):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.model = ui.SimpleStringModel(text)

    def __init__(self, values: list) -> None:
        super().__init__()
        self._items: list = []
        self._current_index = ui.SimpleIntModel(0)
        self._current_index.add_value_changed_fn(lambda _m: self._item_changed(None))
        self.replace_item_list(values)

    def replace_item_list(self, values: list) -> None:
        self._items = [self._StringItem(v) for v in values]
        self._current_index.set_value(0)
        self._item_changed(None)

    def get_selected_value(self) -> str:
        index = self._current_index.get_value_as_int()
        if 0 <= index < len(self._items):
            return self._items[index].model.get_value_as_string()
        return ""

    def get_selected_index(self) -> int:
        return self._current_index.get_value_as_int()

    def add_index_changed_fn(self, fn) -> None:
        self._current_index.add_value_changed_fn(lambda _m: fn())

    def get_item_children(self, item=None):
        return self._items

    def get_item_value_model(self, item=None, column_id: int = 0):
        if item is None:
            return self._current_index
        return item.model


# 주요 액션 버튼(Data collection / Tuning start / Apply to twin / Matching results) 공용 스타일
# — 원본 UI 목업의 민트그린 버튼 재현. 네 버튼 모두 같은 색으로 통일.
_PRIMARY_BUTTON_STYLE = {
    "Button": {"background_color": 0xFF7FD8A0, "border_radius": 4},
    "Button.Label": {"color": 0xFF16211C, "font_size": 14},
    "Button:hovered": {"background_color": 0xFF8FE0B4},
    "Button:pressed": {"background_color": 0xFF6BC090},
}

# Step 3 "매칭 스코어" 산식에 쓰는 휴리스틱 허용 오차(rad).
# mean_error가 0이면 100%, 이 값 이상이면 0% — 실제 기준값은 추후 조정 가능.
_MATCH_SCORE_TOLERANCE_RAD = 0.2

# 튜닝은 물리 텐서 API 도메인(회전 관절은 rad 기준)에서 게인을 다룬다.
# 반면 USD DriveAPI / Property 패널은 각도 드라이브를 degree 기준으로 표시한다
# (Isaac Sim 의 set_gains(save_to_usd=True) 도 이 변환을 해서 써넣는다).
# 표에 Property 패널과 같은 숫자를 보여주려고 rad→deg 로 환산한다.
_RAD_TO_DEG = math.pi / 180.0


def _build_simple_table(headers: list, rows: list) -> None:
    """현재 UI 컨테이너 안에 3열짜리 간단한 표(헤더 + 값 행들)를 렌더링.

    "Optimization parameters"와 "Per-joint error" 두 표가 모양이 같아서 공용으로 뺌.

    주의: 이 함수는 ui.Frame(build_fn=...)의 build_fn으로 쓰인다. ui.Frame은 자식을
    하나만 가질 수 있어서, 헤더/구분선/행들을 감싸는 컨테이너 없이 나열하면 맨 마지막에
    만들어진 위젯 하나만 Frame의 자식으로 남고 그 앞의 것들은 다 사라진다 (실제로
    이 버그 때문에 "Joint 16" 행 하나만 보이는 문제가 있었음). 그래서 전체를 VStack
    하나로 묶어야 한다.
    """
    with ui.VStack(height=0):
        with ui.HStack(height=20):
            for h in headers:
                ui.Label(h, style={"color": 0xFF9B9B9B}, width=ui.Fraction(1))
        ui.Separator(height=2)
        if not rows:
            ui.Label("No data", style={"color": 0xFF7A7A7A})
            return
        for row in rows:
            with ui.HStack(height=22):
                for cell in row:
                    ui.Label(str(cell), width=ui.Fraction(1))


class TwinParamTunerExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        carb.log_info("[Parameter Tuning] on_startup")

        # 기본값 (원본 UI 목업 값과 동일: 333Hz / 3.0s)
        self._default_control_hz = 15.0
        self._default_measurement_length_s = 3.0
        self._default_column_prefix = "qpos_"
        # 모든 결과 CSV를 저장하는 공용 폴더 — q_sim.csv(Data collection),
        # q_real.csv(Data collection, 하드웨어 replay 결과), q_sim_a.csv(Apply to
        # twin, 게인 적용 후 재생 결과), tuned_gains.csv(Step 2 튜닝 결과 게인)
        # 전부 이 폴더 하나에 평평하게(파일명 고정, 매 실행마다 덮어씀) 저장한다.
        # 그 외 부가적인 CSV 저장(비교 CSV 등)은 뺐다.
        self._default_save_dir = ""
        self._default_output_path = os.path.join(self._default_save_dir, "q_sim.csv")
        # Apply to twin이 튜닝된 게인 적용 직후, 원래 q_ref를 다시 재생해서 얻는
        # "적용 후" 궤적 저장 경로. q_sim.csv(Step 1의 튜닝 전/원본 수집 결과)는
        # 덮어쓰지 않고 별도 파일로 둔다.
        self._default_output_path_after_apply = os.path.join(
            self._default_save_dir, "q_sim_a.csv"
        )
        # Step 2 튜닝이 끝나면 best stiffness/damping 을 이 CSV로도 저장한다
        # (USD DriveAPI 단위 = per degree, 'Optimization parameters' 표와 동일).
        # USD 스테이지에만 남던 값을 파일로도 남겨 실행 간 비교/기록이 되게 한다.
        self._default_tuned_gains_path = os.path.join(
            self._default_save_dir, "tuned_gains.csv"
        )
        # CMA-ES 세대별 손실 로그 — 수렴 그래프(세대 vs loss)용.
        self._default_cma_history_path = os.path.join(
            self._default_save_dir, "cma_history.csv"
        )
        # 두 구간 소요시간을 한 파일에 한 줄씩 append 하는 로그:
        #   1) 첫 q_ref(레퍼런스) 파일 선택 시점 -> 'Matching results' 완료 시점
        #   2) 'Tuning start' 버튼 클릭 시점    -> 튜닝 완료 시점
        self._default_timing_log_path = os.path.join(
            self._default_save_dir, "timing_log.csv"
        )

        # Step 1 "real hardware" 옵션 기본값 — 사용자가 CLI로 검증 완료한 값들
        # (python3 ReplayRefTrajectory.py --ref-csv logs/q_ref.csv --control-hz 15
        #  --measure-time 3 --out-dir logs/run2 --comm-wait 0.03).
        # runner command는 실행 환경(Isaac Sim 프로세스)의 PATH에 conda가 없어서
        # ("conda run -n dualarm2026 python3" 시도 시 FileNotFoundError: No such
        # file or directory: 'conda' 로 실제 확인됨) SWRobotics 등 하드웨어
        # 의존성이 설치된 env의 python 인터프리터 절대경로를 직접 쓴다 — conda
        # activate/run을 거치지 않으므로 PATH에 conda가 있는지와 무관하게 항상
        # 동작한다. 그래도 UI에서 바로 고칠 수 있게 필드로 남겨뒀다.
        self._default_hw_script_path = ""
        self._default_hw_runner_cmd = ""
        self._default_hw_port = "/dev/ttyUSB0"
        self._default_hw_comm_wait = 0.03

        self._file_picker = None
        self._file_picker_target_field = None
        self._robot_combo_model = _RobotComboModel()
        self._algorithm_combo_model = _StringListComboModel(["CMA-ES", "PSO", "Random Search"])
        self._loss_combo_model = _StringListComboModel(["MSE", "RMSE", "MAE"])
        self._data_collection_task = None
        self._tuning_task = None
        self._apply_task = None
        self._active_runner = None  # 튜닝 중인 rollout 러너 (deferred teardown 용)
        # 튜닝 진행 중 "Tuning start"를 다시 누르면 True → 현재 세대 끝나고 중단.
        self._tuning_should_stop = False

        # 소요시간 측정용 시작 시각(time.time(), 초). None = 아직 시작 안 함.
        #  _t1: 첫 q_ref 파일 선택 시점 → 'Matching results' 완료까지
        #  _t2: 'Tuning start' 클릭 시점 → 튜닝 완료까지
        self._t1_ref_selected_at = None
        self._t2_tuning_started_at = None

        # Step 2 결과 — 튜닝이 끝나면 채워짐 (best stiffness/damping, 트윈 DOF 순서).
        self._tuned_stiffness = None  # np.ndarray (J,) | None
        self._tuned_damping = None  # np.ndarray (J,) | None
        self._tuned_joint_names = None  # list[str] | None

        # Step 2 기본값 (원본 UI 목업 값과 동일)
        self._default_learning_param_joints = 16  # 기본 학습 관절 수 (Allegro=16 기준, 학습 대상: stiffness, damping)
        self._default_population = 50
        self._default_generations = 100
        self._default_env_count = 128  # physX cloner로 병렬 실행할 환경 수

        # Step 1 상태 — 'Data collection'이 끝나야 채워짐.
        self._last_q_ref = None  # Trajectory | None (Step 1에서 로드한 q_ref, twin에 명령으로 주입)
        self._last_q_sim = None  # Trajectory | None (트윈 실측 q_sim, isaac 이름/순서)
        # Step 2 상태 — Tuning start / Matching results / Apply to twin 중 아무거나
        # 처음 눌렀을 때, 그 시점의 "Reference (q_real)" 필드 경로에서 로드됨.
        self._last_q_real = None  # Trajectory | None (실측 정답 궤적, loss/비교 전용)
        # Step 3 상태 — 'Matching results' 클릭 시 채워짐 (q_real을 q_sim 순서로 재배열한 값 기준).
        self._last_aligned_q_real = None  # np.ndarray (N, J) | None
        self._last_abs_error = None  # np.ndarray (N, J) | None
        self._last_rms_error_over_time = None  # np.ndarray (N,) | None
        # 'Apply to twin' 직전(= 튜닝 전 게인)의 매칭 상태 스냅샷 — before/after 비교용.
        self._baseline_metrics = None  # dict | None
        self._baseline_q_sim = None  # Trajectory | None
        self._verification_joint_combo_model = _StringListComboModel(
            [f"Joint {i + 1}" for i in range(self._default_learning_param_joints)]
        )
        self._verification_joint_combo_model.add_index_changed_fn(
            self._on_verification_joint_changed
        )

        self._window = ui.Window("Parameter Tuning", width=460, height=900)
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
                with ui.VStack(spacing=8, height=0):
                    ui.Label(
                        "Real-to-twin joint trajectory auto-tuning - Mode B (trajectory file playback)",
                        style={"font_size": 11, "color": 0xFF9B9B9B},
                        word_wrap=True,
                    )
                    ui.Spacer(height=4)

                    ui.Label("Robot Selection")
                    ui.ComboBox(self._robot_combo_model, height=24)
                    ui.Spacer(height=4)

                    with ui.CollapsableFrame(
                        "1. Reference trajectory",
                        height=0,
                        collapsed=False,
                        style={"Frame": {"padding": 8}},
                    ):
                        with ui.VStack(spacing=8, height=0):
                            ui.Label(
                                "Load q_ref trajectory file "
                                "(command injected into the twin every step)"
                            )
                            with ui.HStack(spacing=4):
                                self._csv_path_field = ui.StringField(height=24)
                                self._csv_path_field.model.set_value("")
                                ui.Button(
                                    "Browse",
                                    width=70,
                                    height=24,
                                    clicked_fn=self._on_browse_clicked,
                                )

                            # MuJoCo 형식(qpos_/qvel_/ctrl_ 등이 한 CSV에 섞여 있을 때만
                            # prefix가 실제로 쓰임)일 때만 보이게, 아래 _update_csv_prefix_visibility
                            # 에서 동적으로 숨김/표시한다.
                            with ui.VStack(spacing=8, height=0) as self._prefix_section:
                                ui.Label(
                                    "CSV column prefix (MuJoCo Allegro, e.g. qpos_). "
                                    "DG-3F 'fN_jM_deg' CSV auto-detected - prefix ignored.",
                                    word_wrap=True,
                                )
                                self._prefix_field = ui.StringField(height=24)
                                self._prefix_field.model.set_value(self._default_column_prefix)

                            with ui.HStack(spacing=8):
                                with ui.VStack(spacing=4):
                                    ui.Label("Control cycle (Hz)")
                                    self._hz_field = ui.FloatField(height=24)
                                    self._hz_field.model.set_value(self._default_control_hz)
                                with ui.VStack(spacing=4):
                                    ui.Label("Measurement length (s)")
                                    self._len_field = ui.FloatField(height=24)
                                    self._len_field.model.set_value(
                                        self._default_measurement_length_s
                                    )

                            ui.Spacer(height=4)
                            with ui.HStack(spacing=6, height=0):
                                self._show_viewport_cb = ui.CheckBox(width=0)
                                self._show_viewport_cb.model.set_value(False)
                                ui.Label(
                                    "Show hand motion in viewport during playback\n"
                                    "(URDF-import assets only; not dg3f.usd)",
                                    style={"color": 0xFF9B9B9B},
                                )

                            ui.Spacer(height=4)
                            ui.Separator(height=2)
                            with ui.HStack(spacing=6, height=0):
                                self._hw_enable_cb = ui.CheckBox(width=0)
                                self._hw_enable_cb.model.set_value(False)
                                ui.Label(
                                    "Also drive real hardware (Delto DG-3F) via "
                                    "ReplayRefTrajectory.py and record q_real\n"
                                    "(moves the physical hand, using the same q_ref/"
                                    "control cycle/length above - DG-3F CSV only)",
                                    style={"color": 0xFF9B9B9B},
                                    word_wrap=True,
                                )
                            with ui.VStack(spacing=8, height=0) as self._hw_options_section:
                                ui.Label("Hardware control script path (ReplayRefTrajectory.py)")
                                self._hw_script_field = ui.StringField(height=24)
                                self._hw_script_field.model.set_value(
                                    self._default_hw_script_path
                                )

                                ui.Label(
                                    "Runner command (prefix before the script, e.g. a "
                                    "conda env's python). Edit this if 'conda' isn't on "
                                    "PATH where Isaac Sim runs.",
                                    word_wrap=True,
                                )
                                self._hw_runner_field = ui.StringField(height=24)
                                self._hw_runner_field.model.set_value(
                                    self._default_hw_runner_cmd
                                )

                                with ui.HStack(spacing=8):
                                    with ui.VStack(spacing=4):
                                        ui.Label("Serial port")
                                        self._hw_port_field = ui.StringField(height=24)
                                        self._hw_port_field.model.set_value(
                                            self._default_hw_port
                                        )
                                    with ui.VStack(spacing=4):
                                        ui.Label("Comm wait (s)")
                                        self._hw_comm_wait_field = ui.FloatField(height=24)
                                        self._hw_comm_wait_field.model.set_value(
                                            self._default_hw_comm_wait
                                        )

                            ui.Spacer(height=4)
                            ui.Button(
                                "Data collection",
                                height=32,
                                clicked_fn=self._on_data_collection_clicked,
                                style=_PRIMARY_BUTTON_STYLE,
                            )

                    ui.Spacer(height=4)

                    with ui.CollapsableFrame(
                        "2. Tuning",
                        height=0,
                        collapsed=False,
                        style={"Frame": {"padding": 8}},
                    ):
                        with ui.VStack(spacing=8, height=0):
                            ui.Label(
                                "Reference (q_real) trajectory file "
                                "- real hand measurement, compared against q_sim for loss"
                            )
                            with ui.HStack(spacing=4):
                                self._q_real_csv_path_field = ui.StringField(height=24)
                                self._q_real_csv_path_field.model.set_value("")
                                ui.Button(
                                    "Browse",
                                    width=70,
                                    height=24,
                                    clicked_fn=self._on_browse_q_real_clicked,
                                )
                            ui.Spacer(height=4)

                            ui.Label("Algorithm")
                            ui.ComboBox(self._algorithm_combo_model, height=24)

                            ui.Label("Learning parameters")
                            with ui.HStack(spacing=8):
                                self._learning_params_field = ui.IntField(width=90, height=24)
                                self._learning_params_field.model.set_value(
                                    self._default_learning_param_joints
                                )
                                ui.Label(
                                    "joints (stiffness, damping)",
                                    style={"color": 0xFF9B9B9B},
                                )

                            with ui.HStack(spacing=8):
                                with ui.VStack(spacing=4):
                                    ui.Label("Population")
                                    self._population_field = ui.IntField(height=24)
                                    self._population_field.model.set_value(
                                        self._default_population
                                    )
                                with ui.VStack(spacing=4):
                                    ui.Label("Generations")
                                    self._generations_field = ui.IntField(height=24)
                                    self._generations_field.model.set_value(
                                        self._default_generations
                                    )
                                    self._generations_field.model.add_value_changed_fn(
                                        lambda _m: self._update_tuning_progress_labels()
                                    )

                            ui.Label("Env (physX cloner)")
                            self._env_field = ui.IntField(height=24)
                            self._env_field.model.set_value(self._default_env_count)

                            ui.Label("Loss indicator")
                            ui.ComboBox(self._loss_combo_model, height=24)

                            ui.Spacer(height=4)
                            ui.Button(
                                "Tuning start",
                                height=36,
                                clicked_fn=self._on_tuning_start_clicked,
                                style=_PRIMARY_BUTTON_STYLE,
                            )

                            ui.Separator(height=8)
                            self._tuning_progress_model = ui.SimpleFloatModel(0.0)
                            ui.ProgressBar(
                                self._tuning_progress_model,
                                height=6,
                                style={
                                    "ProgressBar": {
                                        "color": 0xFF7FD8A0,
                                        "background_color": 0xFF2A2A2A,
                                        "border_radius": 3,
                                    }
                                },
                            )
                            with ui.HStack(height=0):
                                self._generation_label = ui.Label(
                                    f"Generation 0 / {self._default_generations}",
                                    style={"color": 0xFF9B9B9B},
                                )
                                ui.Spacer()
                                ui.Label("best loss ", style={"color": 0xFF9B9B9B})
                                self._best_loss_label = ui.Label(
                                    "-", style={"color": 0xFF7FD8A0}
                                )

                    ui.Spacer(height=4)

                    with ui.CollapsableFrame(
                        "3. Verification",
                        height=0,
                        collapsed=False,
                        style={"Frame": {"padding": 8}},
                    ):
                        with ui.VStack(spacing=8, height=0):
                            ui.Label("Optimization parameters")
                            with ui.ScrollingFrame(
                                height=140,
                                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                style={"Frame": {"background_color": 0xFF1E1E1E}},
                            ):
                                self._params_table_frame = ui.Frame(
                                    build_fn=self._build_params_table
                                )

                            ui.Spacer(height=4)
                            ui.Button(
                                "Apply to twin",
                                height=32,
                                clicked_fn=self._on_apply_to_twin_clicked,
                                style=_PRIMARY_BUTTON_STYLE,
                            )
                            ui.Button(
                                "Matching results",
                                height=36,
                                clicked_fn=self._on_matching_results_clicked,
                                style=_PRIMARY_BUTTON_STYLE,
                            )

                            ui.Separator(height=8)
                            with ui.HStack(height=0):
                                ui.Label("Match score", style={"color": 0xFF9B9B9B})
                                ui.Spacer()
                                self._match_score_label = ui.Label(
                                    "-", style={"color": 0xFF7FD8A0, "font_size": 22}
                                )

                            with ui.HStack(spacing=8, height=56):
                                with ui.Frame(
                                    style={
                                        "Frame": {
                                            "background_color": 0xFF232323,
                                            "border_radius": 4,
                                            "padding": 8,
                                        }
                                    }
                                ):
                                    with ui.VStack(spacing=2):
                                        ui.Label("Mean error", style={"color": 0xFF9B9B9B})
                                        self._mean_error_label = ui.Label("-")
                                with ui.Frame(
                                    style={
                                        "Frame": {
                                            "background_color": 0xFF232323,
                                            "border_radius": 4,
                                            "padding": 8,
                                        }
                                    }
                                ):
                                    with ui.VStack(spacing=2):
                                        ui.Label("Max error", style={"color": 0xFF9B9B9B})
                                        self._max_error_label = ui.Label("-")

                            ui.Spacer(height=4)
                            ui.Label("Per-joint error")
                            with ui.ScrollingFrame(
                                height=120,
                                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                style={"Frame": {"background_color": 0xFF1E1E1E}},
                            ):
                                self._joint_error_table_frame = ui.Frame(
                                    build_fn=self._build_joint_error_table
                                )

                            ui.Spacer(height=4)
                            with ui.HStack(spacing=8, height=24):
                                ui.Label("Joint tracking", width=0)
                                ui.ComboBox(self._verification_joint_combo_model, height=24)
                            self._joint_overlay_plot_frame = ui.Frame(
                                height=130, build_fn=self._build_joint_overlay_plot
                            )

                            ui.Spacer(height=4)
                            ui.Label("Tracking error over time (RMS across joints, rad)")
                            self._error_over_time_plot_frame = ui.Frame(
                                height=110, build_fn=self._build_error_over_time_plot
                            )

                    ui.Separator(height=8)
                    self._status_label = ui.Label(
                        "Idle.", word_wrap=True, style={"color": 0xFFB0B0B0}
                    )

        self._on_refresh_robot_list_clicked()

        # q_ref/q_real 경로나 prefix 값이 바뀔 때마다, 실제로 prefix가 쓰이는
        # MuJoCo 형식일 때만 prefix 입력칸이 보이게 갱신.
        self._csv_path_field.model.add_value_changed_fn(
            lambda _m: self._update_csv_prefix_visibility()
        )
        self._q_real_csv_path_field.model.add_value_changed_fn(
            lambda _m: self._update_csv_prefix_visibility()
        )
        self._prefix_field.model.add_value_changed_fn(
            lambda _m: self._update_csv_prefix_visibility()
        )
        self._update_csv_prefix_visibility()

        # 첫 구간 소요시간의 시작점: q_ref(레퍼런스) 경로가 채워지는 시점을 기록.
        # Browse 로 고르면 한 번, 직접 타이핑하면 마지막 입력 시점이 시작점이 된다.
        self._csv_path_field.model.add_value_changed_fn(
            lambda _m: self._on_q_ref_path_changed()
        )

        # 하드웨어 옵션 필드들은 체크박스를 켰을 때만 보여준다 (기본 꺼짐 — 실제
        # 손을 움직이는 옵션이라 실수로 켜져 있는 채로 두지 않게).
        self._hw_options_section.visible = False
        self._hw_enable_cb.model.add_value_changed_fn(
            lambda m: setattr(self._hw_options_section, "visible", m.get_value_as_bool())
        )

    def _update_csv_prefix_visibility(self) -> None:
        """q_ref(Step 1)/q_real(Step 2) 중 하나라도 MuJoCo 형식(prefix가 실제로
        구분에 쓰이는 형식)으로 인식되면 prefix 입력칸을 보여주고, 둘 다 DG-3F나
        일반 CSV(=prefix 무시됨)면 숨긴다. 아직 경로가 비어 있거나 파일을 못
        읽으면(예: 타이핑 중) 판단 불가 상태로 보고 안전하게 보여준다.
        """
        column_prefix = (
            self._prefix_field.model.get_value_as_string().strip()
            or self._default_column_prefix
        )
        paths = [
            self._csv_path_field.model.get_value_as_string().strip(),
            self._q_real_csv_path_field.model.get_value_as_string().strip(),
        ]
        existing_paths = [p for p in paths if p]

        show = not existing_paths  # 아직 아무 경로도 없으면 기본값: 보여줌
        for p in existing_paths:
            try:
                if detect_csv_kind(p, column_prefix=column_prefix) == "mujoco":
                    show = True
            except Exception:  # noqa: BLE001 - 판단 불가(예: 파일 없음) → 안전하게 보여줌
                show = True

        self._prefix_section.visible = show

    def _on_q_ref_path_changed(self) -> None:
        """q_ref(레퍼런스) 경로 필드가 바뀔 때 호출 — 첫 구간 소요시간의 시작
        시각을 기록한다. 경로가 비면 시작 시각도 지운다.
        """
        path = self._csv_path_field.model.get_value_as_string().strip()
        self._t1_ref_selected_at = time.time() if path else None

    def _append_timing_log(self, measurement: str, seconds: float, detail: str = "") -> None:
        """두 구간 소요시간을 timing_log.csv 한 파일에 한 줄씩 append.

        컬럼: finished_at(측정 종료 시각, ISO), measurement(구간 이름),
              seconds(소요 초), detail(부가 설명). 파일이 없으면 헤더를 먼저 쓴다.
        저장 실패는 기능 자체를 막지 않도록 로그만 남기고 삼킨다(다른 CSV 저장과 동일).
        """
        path = self._default_timing_log_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            new_file = not os.path.isfile(path)
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["finished_at", "measurement", "seconds", "detail"])
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                        measurement,
                        f"{seconds:.3f}",
                        detail,
                    ]
                )
            carb.log_info(
                f"[Parameter Tuning] timing logged: {measurement} = {seconds:.3f}s "
                f"-> {path}"
            )
        except Exception as e:  # noqa: BLE001 - 로깅 실패가 기능을 막지 않게
            carb.log_warn(f"[Parameter Tuning] could not write timing_log.csv: {e}")

    def _on_refresh_robot_list_clicked(self) -> None:
        stage = omni.usd.get_context().get_stage()
        prim_paths: list = []
        if stage:
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    continue
                path = prim.GetPath()
                # ArticulationRootAPI 가 조인트 프림에 걸려 있으면(예: DG-3F 의
                # 고정 root_joint /dg3f_b/root_joint), Isaac 의 SingleArticulation
                # 은 보통 그 상위 바디/Xform 경로를 원한다 → 조인트면 부모로 올린다.
                if prim.IsA(UsdPhysics.Joint):
                    parent = prim.GetParent()
                    if parent and parent.IsValid() and not parent.IsPseudoRoot():
                        path = parent.GetPath()
                p = str(path)
                if p not in prim_paths:
                    prim_paths.append(p)

        # 실제로 스테이지에서 찾은 prim만 목록에 올린다 — 아무것도 없을 때 하드코딩된
        # 기본 경로를 마치 존재하는 로봇인 것처럼 보여주지 않는다 (그 경로는 실존하지
        # 않으므로, 그 상태로 Data collection/Tuning start를 누르면 다른 곳에서
        # 알아보기 힘든 에러로 이어짐). 없으면 콤보모델이 선택 불가 안내 문구를 보여준다.
        self._robot_combo_model.replace_item_list(prim_paths)

    def _on_browse_clicked(self) -> None:
        self._show_file_picker(self._csv_path_field)

    def _on_browse_q_real_clicked(self) -> None:
        self._show_file_picker(self._q_real_csv_path_field)

    def _show_file_picker(self, target_field) -> None:
        """target_field 를 매번 갱신해서 두 CSV 필드(q_ref/q_real)가 파일 피커
        하나를 같이 쓰게 한다 — on_select 는 self._file_picker_target_field 를
        호출 시점에 읽으므로, 다이얼로그를 최초 1회만 만들어도 대상이 바뀐다."""
        self._file_picker_target_field = target_field

        def on_select(filename: str, dirname: str) -> None:
            full_path = f"{dirname.rstrip('/')}/{filename}" if dirname else filename
            self._file_picker_target_field.model.set_value(full_path)
            self._file_picker.hide()

        def on_cancel(filename: str, dirname: str) -> None:
            self._file_picker.hide()

        if self._file_picker is None:
            self._file_picker = FilePickerDialog(
                "Select trajectory file",
                allow_multi_selection=False,
                apply_button_label="Select",
                click_apply_handler=lambda filename, dirname: on_select(filename, dirname),
                click_cancel_handler=lambda filename, dirname: on_cancel(filename, dirname),
                file_extension_options=[(".csv", "CSV Files (*.csv)"), ("", "All Files (*.*)")],
            )
        self._file_picker.show()

    def _on_data_collection_clicked(self) -> None:
        prim_path = self._robot_combo_model.get_selected_path()
        csv_path = self._csv_path_field.model.get_value_as_string().strip()
        column_prefix = self._prefix_field.model.get_value_as_string().strip()
        control_hz = self._hz_field.model.get_value_as_float()
        measurement_length_s = self._len_field.model.get_value_as_float()
        show_in_viewport = self._show_viewport_cb.model.get_value_as_bool()

        hw_enabled = self._hw_enable_cb.model.get_value_as_bool()
        hw_script_path = self._hw_script_field.model.get_value_as_string().strip()
        hw_runner_cmd = self._hw_runner_field.model.get_value_as_string().strip()
        hw_port = self._hw_port_field.model.get_value_as_string().strip()
        hw_comm_wait = self._hw_comm_wait_field.model.get_value_as_float()

        if not prim_path:
            self._set_status("Error: select a robot in Robot Selection", error=True)
            return
        if not csv_path:
            self._set_status("Error: q_ref CSV path must be set", error=True)
            return

        if hw_enabled:
            if not hw_script_path:
                self._set_status(
                    "Error: hardware control script path must be set", error=True
                )
                return
            if not hw_runner_cmd:
                self._set_status("Error: hardware runner command must be set", error=True)
                return
            # ReplayRefTrajectory.py 는 DG-3F 전용(t_s, fN_jM_deg 컬럼) CSV만 읽을 수
            # 있다 — 트윈이 MuJoCo/Allegro 형식 q_ref를 쓰는 중이면 여기서 미리 막는다
            # (그대로 두면 하드웨어 프로세스가 한참 뒤 파싱 에러로 실패함).
            try:
                kind = detect_csv_kind(csv_path, column_prefix=column_prefix)
            except Exception as e:  # noqa: BLE001
                self._set_status(
                    f"Error: cannot read q_ref CSV for hardware check: {e}", error=True
                )
                return
            if kind != "dg3f":
                self._set_status(
                    "Error: real hardware collection needs a DG-3F q_ref CSV "
                    "(t_s, f1_j1_deg.. columns) - ReplayRefTrajectory.py only supports "
                    f"that format. Detected: {kind}.",
                    error=True,
                )
                return

        self._set_status(
            "Collecting... twin (q_sim) + real hardware (q_real) together"
            if hw_enabled
            else "Collecting... (also check the Isaac Sim console log)"
        )

        # 999 스텝 재생을 버튼 콜백 안에서 동기적으로 다 돌리면, sim.step(render=True)가
        # 내부에서 반복 호출하는 app.update()가 이미 진행 중인 app.update() 프레임
        # 안에서 재귀적으로 실행돼 렌더러 커맨드 리스트가 깨지고 크래시로 이어진다.
        # asyncio task로 분리해서 매 스텝 사이에 Kit 프레임 루프로 제어권을 돌려준다.
        self._data_collection_task = asyncio.ensure_future(
            self._collect_data_async(
                prim_path,
                csv_path,
                column_prefix,
                control_hz,
                measurement_length_s,
                show_in_viewport,
                hw_enabled=hw_enabled,
                hw_script_path=hw_script_path,
                hw_runner_cmd=hw_runner_cmd,
                hw_port=hw_port,
                hw_comm_wait=hw_comm_wait,
            )
        )

    async def _collect_data_async(
        self,
        prim_path: str,
        csv_path: str,
        column_prefix: str,
        control_hz: float,
        measurement_length_s: float,
        show_in_viewport: bool = False,
        hw_enabled: bool = False,
        hw_script_path: str = "",
        hw_runner_cmd: str = "",
        hw_port: str = "",
        hw_comm_wait: float = 0.02,
    ) -> None:
        # try 블록 진입 전에 미리 None으로 초기화 — q_ref 로드 자체가 실패해서
        # 아래 try 안의 대입문에 도달하기 전에 예외가 나도, except 블록에서
        # `hw_handle`을 참조할 때 NameError가 나지 않게 한다(하드웨어를 아직
        # 시작조차 안 했다는 뜻이므로 None이 맞다).
        hw_handle = None
        try:
            num_steps = expected_num_samples(control_hz, measurement_length_s)

            q_ref = load_hand_trajectory_csv(
                csv_path, column_prefix=column_prefix, num_samples=num_steps
            )
            csv_format = describe_csv_format(csv_path, column_prefix=column_prefix)
            carb.log_info(
                f"[Parameter Tuning] q_ref loaded: "
                f"{q_ref.num_samples} samples x {q_ref.num_joints} joints, "
                f"joints={q_ref.joint_names}, csv_format={csv_format}"
            )

            # 하드웨어 replay는 트윈 rollout의 asyncio Task와 "동시에 스케줄된
            # 별도 asyncio Task"로는 절대 만들지 않는다 (asyncio.gather나
            # run_in_executor로 감싸는 것도 포함) — 둘 다 시도해봤는데 Kit
            # (omni.kit.async_engine)의 asyncio 루프가 PhysX 스텝 도중 재진입돼
            # 아래 오류로 깨지는 게 재현됐다:
            #   RuntimeError: Cannot enter into task <hw task> while another
            #   task <run_trajectory_on_twin task> is being executed.
            #   IndexError: pop from an empty deque
            # (asyncio.Task.__step()의 loop당 "현재 실행 중 task" 가드가, 다른
            # OS 스레드의 완료 콜백이 PhysX 스텝 도중 끼어들면서 깨지는 것으로
            # 보인다.) 그래서 대신: 하드웨어 서브프로세스는 지금 이 자리에서
            # (동기적으로, 논블로킹인) subprocess.Popen + 순수 threading.Thread로
            # "그냥 지금 시작"만 해두고 즉시 리턴 — 이 스레드는 asyncio를 전혀
            # 건드리지 않으므로 트윈 rollout이 도는 동안 Kit 루프와 아무 것도
            # 경쟁하지 않는다. 두 동작 다 같은 q_ref/control_hz/measurement_length_s
            # 로 "실제 시간(wall-clock)" 상으로는 겹쳐서 돌기 때문에(OS 레벨의
            # 진짜 동시 실행), 실측 시점은 여전히 동시다 — 다만 이 asyncio
            # 태스크(트윈)가 끝날 때까지는 하드웨어 스레드의 "완료"를 이 함수가
            # 들여다보지 않는다(트윈이 끝난 뒤 아래에서 안전하게 폴링).
            if hw_enabled:
                hw_handle = self._start_hardware_replay_thread(
                    csv_path=csv_path,
                    control_hz=control_hz,
                    measurement_length_s=measurement_length_s,
                    script_path=hw_script_path,
                    runner_cmd=hw_runner_cmd,
                    port=hw_port,
                    comm_wait=hw_comm_wait,
                )

            q_sim = await run_trajectory_on_twin(
                articulation_prim_path=prim_path,
                q_ref=q_ref,
                control_hz=control_hz,
                measurement_length_s=measurement_length_s,
                show_in_viewport=show_in_viewport,
            )

            save_trajectory_csv(
                self._default_output_path,
                q_sim.joint_names,
                q_sim.timestamps,
                q_sim.positions,
            )

            # Step 2/3에서 loss/비교에 쓸 수 있도록 저장.
            # 새로 수집했으니 이전 매칭 결과(오차/그래프)는 무효 — 캐시를 비워서
            # Matching results를 다시 눌러야 최신 값으로 갱신되게 한다.
            # 수동 재수집은 새 측정 사이클이므로 before/after 기준(baseline)도 초기화.
            self._last_q_ref = q_ref
            self._last_q_sim = q_sim
            self._last_aligned_q_real = None
            self._last_abs_error = None
            self._last_rms_error_over_time = None
            self._baseline_metrics = None
            self._baseline_q_sim = None
            self._verification_joint_combo_model.replace_item_list(q_sim.joint_names)
            # 트윈 DOF 수보다 'Learning parameters' 가 크면 튜닝이 에러를 낸다
            # (Allegro=16 기본값 vs DG-3F=12 등). 넘칠 때만 DOF 수로 낮춘다.
            if self._learning_params_field.model.get_value_as_int() > q_sim.num_joints:
                self._learning_params_field.model.set_value(q_sim.num_joints)
            self._params_table_frame.rebuild()
            self._joint_error_table_frame.rebuild()
            self._joint_overlay_plot_frame.rebuild()
            self._error_over_time_plot_frame.rebuild()

            # 하드웨어 replay 결과 반영. 트윈 rollout(asyncio Task)이 이미 끝난
            # 뒤이므로, 이제는 다른 task와 경쟁할 위험 없이 안전하게 폴링해서
            # 기다릴 수 있다. 보통 이 시점엔 측정 구간 자체는 끝나 있고, 하드웨어
            # 쪽 ramp-in/통신 오버헤드/시작 자세 복귀만 조금 더 남아있는 정도라
            # 오래 걸리지 않는다.
            hw_note = ""
            if hw_handle is not None:
                self._set_status(
                    f"q_sim done ({q_sim.num_samples} samples). Waiting for real "
                    "hardware replay to finish..."
                )
                hw_note = await self._finish_hardware_replay_and_note(hw_handle)

            self._set_status(
                f"Done. Saved q_sim ({q_sim.num_samples} samples) to: "
                f"{self._default_output_path} | q_ref CSV: {csv_format}{hw_note}"
            )

        except asyncio.CancelledError:
            # 취소(익스텐션 종료 등)는 여기서 하드웨어를 기다리지 않고 그대로
            # 전파한다 — 트윈이 취소된 시점에 하드웨어가 아직 돌고 있었다면,
            # 백그라운드 스레드/서브프로세스는 데몬으로 남아 자기 안전 복귀
            # 로직까지 스스로 끝까지 진행한다(강제로 끊는 것보다 안전). 이미
            # _wait_for_hardware_replay 안에 있었을 때의 취소는 거기서 별도로
            # proc.terminate()까지 처리한다.
            raise
        except Exception as e:  # noqa: BLE001 - UI callback, so surface the error on screen
            carb.log_error(f"[Parameter Tuning] Step1 failed: {e}")
            hw_note = ""
            if hw_handle is not None:
                # 트윈이 실패해도 이미 시작된 하드웨어 replay(실제 손 동작)는
                # 계속 진행 중일 수 있다 — 방치하지 않고 끝까지(자체 안전 복귀
                # 포함) 기다렸다가 결과를 같이 보고한다.
                self._set_status(
                    f"Error: {e} | waiting for real hardware replay to finish "
                    "before reporting...",
                    error=True,
                )
                hw_note = await self._finish_hardware_replay_and_note(hw_handle)
            self._set_status(f"Error: {e}{hw_note}", error=True)

    async def _finish_hardware_replay_and_note(self, hw_handle: dict) -> str:
        """`_wait_for_hardware_replay()`를 호출하고, 성공하면 q_real.csv를 Step 2
        필드에 채워 로드까지 마친 뒤, 상태줄에 붙일 요약 문구를 돌려준다.
        (성공/실패 케이스를 성공 경로와 실패 경로 양쪽에서 똑같이 처리하려고
        `_collect_data_async`에서 공용으로 뺐다.) CancelledError는 그대로
        전파한다(호출부가 처리).
        """
        try:
            real_csv_path = await self._wait_for_hardware_replay(hw_handle)
        except asyncio.CancelledError:
            raise
        except Exception as hw_exc:  # noqa: BLE001 - reported, not fatal to caller
            carb.log_error(f"[Parameter Tuning] hardware q_real collection failed: {hw_exc}")
            return f" | Hardware q_real FAILED: {hw_exc}"

        self._q_real_csv_path_field.model.set_value(real_csv_path)
        q_real = self._load_q_real_from_field()
        if q_real is not None:
            return f" | q_real ({q_real.num_samples} samples) saved to: {real_csv_path}"
        return (
            f" | Hardware q_real saved to {real_csv_path} but failed to load "
            "(see error above)"
        )

    def _start_hardware_replay_thread(
        self,
        csv_path: str,
        control_hz: float,
        measurement_length_s: float,
        script_path: str,
        runner_cmd: str,
        port: str,
        comm_wait: float,
    ) -> dict:
        """실제 Delto DG-3F 손에 같은 q_ref를 재생하는 ReplayRefTrajectory.py를
        지금 바로(논블로킹) 별도 OS 프로세스로 띄우고 handle을 반환한다.

        *** asyncio Task로 감싸지 않는 이유 (중요 — 실제로 크래시가 재현됨) ***
        이전 구현은 asyncio.create_subprocess_exec, 그리고 그다음엔
        loop.run_in_executor로 감싼 하드웨어 작업을 트윈 rollout
        (run_trajectory_on_twin, 그 자체가 이미 asyncio Task) 과 asyncio.gather
        로 "나란히 실행되는 두 번째 asyncio Task"로 만들었었다. 두 경우 모두
        Isaac Sim 콘솔에서 PhysX 스텝 도중 아래 오류로 Kit의 asyncio 루프
        (omni.kit.async_engine)가 깨지는 게 재현됐다:
            RuntimeError: Cannot enter into task <hw task> while another task
            <run_trajectory_on_twin task> is being executed.
            IndexError: pop from an empty deque
        (asyncio.Task.__step()의 loop당 "현재 실행 중 task" 재진입 가드
        (_current_tasks)가, 트윈 태스크의 스텝이 아직 안 끝난 상태에서 다른
        스레드의 완료 통지로 두 번째 task의 스텝이 끼어들며 깨지는 것으로
        보인다 — 이 Isaac Sim/Kit 빌드에서 두 번 다른 방식으로 재현됨.)

        그래서: 하드웨어 replay는 Kit의 asyncio 루프에 "동시에 스케줄된 별도
        task"로 절대 등록하지 않는다. 이 메서드는 subprocess.Popen()과
        threading.Thread.start()만 호출하고(둘 다 즉시 리턴하는 논블로킹 호출)
        바로 반환한다 — 만들어진 스레드는 asyncio를 전혀 건드리지 않고 그냥
        구독 없이 백그라운드에서 실제 손을 움직이며 로그만 찍는다. 호출부는
        이 직후 `await run_trajectory_on_twin(...)`를 도는 동안 이 스레드를
        전혀 들여다보지 않고, 그 await가 끝난 뒤(더 이상 경쟁할 다른 task가
        없을 때)에만 `_wait_for_hardware_replay()`로 폴링해서 기다린다. 두
        동작 다 같은 q_ref/control_hz/measurement_length_s로 실제 시간
        (wall-clock) 상으로는 겹쳐서 돌므로(OS 프로세스 레벨의 진짜 동시
        실행), 실측 구간 자체는 여전히 "동시"다.

        검증 완료된 호출 형태(사용자 CLI 테스트)와 동일하게 구성한다:
            <runner_cmd> <script_path> --ref-csv <csv_path> --control-hz <hz>
                --measure-time <len> --out-dir <out_dir> --comm-wait <comm_wait>
                --port <port> --yes

        반환: {"thread": Thread|None, "out_dir": str|None, "result": dict,
               "launch_error": Exception|None}
        launch_error가 있으면 스레드를 아예 시작하지 못한 것 — thread는 None.
        """
        script_path = os.path.abspath(os.path.expanduser(script_path))
        if not os.path.isfile(script_path):
            return {
                "thread": None,
                "out_dir": None,
                "result": {},
                "launch_error": FileNotFoundError(
                    f"Hardware control script not found: {script_path}"
                ),
            }
        script_dir = os.path.dirname(script_path)

        try:
            runner_parts = shlex.split(runner_cmd)
        except ValueError as e:
            return {
                "thread": None,
                "out_dir": None,
                "result": {},
                "launch_error": ValueError(f"Invalid runner command '{runner_cmd}': {e}"),
            }
        if not runner_parts:
            return {
                "thread": None,
                "out_dir": None,
                "result": {},
                "launch_error": ValueError("Runner command must not be empty"),
            }

        # q_sim.csv/q_sim_a.csv와 같은 공용 폴더에 평평하게 저장 — 매 실행마다
        # q_real.csv(그리고 ReplayRefTrajectory.py 자체 로그인 q_ref.csv)를
        # 덮어쓴다. (ReplayRefTrajectory.py의 TrajectoryCsvLogger가 out_dir을
        # 알아서 만들어주므로 여기서 따로 mkdir할 필요는 없다.)
        out_dir = self._default_save_dir

        cmd = runner_parts + [
            script_path,
            "--ref-csv", os.path.abspath(os.path.expanduser(csv_path)),
            "--control-hz", str(control_hz),
            "--measure-time", str(measurement_length_s),
            "--out-dir", out_dir,
            "--comm-wait", str(comm_wait),
            "--port", port,
            "--yes",  # 비대화형 실행 - 확인 프롬프트에서 멈추지 않게
        ]
        carb.log_info(f"[Parameter Tuning] launching hardware replay: {' '.join(cmd)}")

        # result: 백그라운드 스레드 <-> 메인(Kit) 스레드 간 결과 전달용. 값들은
        # 스레드 쪽에서 한 번씩만 쓰고(write-once), 메인 쪽에서는 스레드가 끝난
        # 뒤(is_alive()==False로 확인 후)에만 읽으므로 별도 락 없이 안전하다.
        # "proc"는 취소(CancelledError) 시 강제 종료용으로 조기에 채워 넣는다.
        result: dict = {"returncode": None, "error": None, "proc": None}

        def _run_and_read() -> None:
            # *** 이 함수는 순수 threading.Thread에서 돈다 — asyncio, Kit,
            # omni.* API를 여기서 절대 호출하지 않는다(carb.log_info 제외 —
            # Carbonite 로깅은 멀티스레드 엔진 전반에서 쓰이는 스레드-세이프
            # API라 여기서도 안전하다). ***
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as e:
                result["error"] = RuntimeError(
                    f"Failed to launch hardware runner command {runner_parts!r}: {e}. "
                    "Check the 'Runner command' field (e.g. 'conda' may not be on PATH "
                    "where Isaac Sim runs - try an absolute python path instead)."
                )
                return
            result["proc"] = proc
            try:
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip()
                        if line:
                            carb.log_info(f"[Parameter Tuning][hw] {line}")
                result["returncode"] = proc.wait()
            except Exception as e:  # noqa: BLE001 - 백그라운드 스레드, 반드시 삼켜야 함
                result["error"] = e

        thread = threading.Thread(
            target=_run_and_read, name="twin_tuner_hw_replay", daemon=True
        )
        thread.start()
        return {"thread": thread, "out_dir": out_dir, "result": result, "launch_error": None}

    async def _wait_for_hardware_replay(self, handle: dict) -> str:
        """`_start_hardware_replay_thread()`가 반환한 handle을 기다려 q_real.csv
        경로를 반환한다. **반드시 `run_trajectory_on_twin`의 await가 완전히
        끝난 뒤에만 호출할 것** — 그 전에는 경쟁하는 asyncio task가 생겨 Kit
        루프가 깨질 수 있다(위 `_start_hardware_replay_thread` docstring 참고).

        폴링에 표준 `asyncio.sleep()`을 썼더니 실제로
        "RuntimeError: no running event loop"가 발생했다 — run_trajectory_on_twin
        이 `app.next_update_async()`(Kit 네이티브 프레임 대기)로만 이어온 코루틴
        컨텍스트를 이어받은 시점이라, asyncio.sleep()이 내부적으로 의존하는
        `asyncio.events.get_running_loop()`이 이 Kit 빌드에서는 기대와 다르게
        실패하는 것으로 보인다. 그래서 표준 asyncio 타이머 기반 sleep 대신,
        run_trajectory_on_twin이 이미 안전하게 쓰고 있는 것과 똑같은 Kit
        네이티브 프레임 대기(app.next_update_async())로 폴링한다 — raw asyncio
        타이머 API를 이 함수 안에서 아예 쓰지 않는다.
        """
        if handle.get("launch_error") is not None:
            raise handle["launch_error"]

        thread: threading.Thread = handle["thread"]
        result: dict = handle["result"]

        # Isaac Sim 런타임 안에서만 import 가능 (run_trajectory_on_twin과 동일한 이유).
        import omni.kit.app

        app = omni.kit.app.get_app()
        try:
            while thread.is_alive():
                await app.next_update_async()
        except asyncio.CancelledError:
            proc = result.get("proc")
            if proc is not None:
                carb.log_warn(
                    "[Parameter Tuning] cancelling hardware replay subprocess "
                    "(extension shutdown/task cancel) - hand may not return home cleanly"
                )
                proc.terminate()
            raise
        thread.join()  # 이미 끝났으니 즉시 반환 - 안전망

        if result.get("error") is not None:
            raise result["error"]
        returncode = result.get("returncode")
        if returncode != 0:
            raise RuntimeError(
                f"ReplayRefTrajectory.py exited with code {returncode} "
                "(see Isaac Sim console log above, lines prefixed [hw])"
            )

        real_csv_path = os.path.join(handle["out_dir"], "q_real.csv")
        if not os.path.isfile(real_csv_path):
            raise FileNotFoundError(
                f"Hardware script finished but did not produce q_real.csv at {real_csv_path}"
            )
        return real_csv_path

    def _update_tuning_progress_labels(self) -> None:
        """Generations 필드 값이 바뀌면 idle 상태의 'Generation 0 / N' 표시도 같이 갱신."""
        generations = self._generations_field.model.get_value_as_int()
        self._generation_label.text = f"Generation 0 / {generations}"

    def _load_q_real_from_field(self):
        """Step 2 'Reference (q_real)' 필드 경로에서 실측 정답 궤적을 로드.

        Tuning start / Matching results / Apply to twin 이 공유 — 클릭 시점마다
        새로 읽어서 self._last_q_real 에 캐시한다(필드를 고쳐도 다음 클릭에 바로
        반영되도록). control_hz/measurement_length_s/column prefix 는 Step 1과
        같은 필드를 그대로 재사용 — q_ref/q_sim/q_real 모두 같은 제어 주기·길이여야
        스텝별로 1:1 비교가 된다.

        성공하면 로드한 Trajectory를 반환하고 self._last_q_real 도 갱신. 실패하면
        상태줄에 에러를 띄우고 None을 반환(caller는 그대로 return하면 됨).
        """
        q_real_path = self._q_real_csv_path_field.model.get_value_as_string().strip()
        if not q_real_path:
            self._set_status(
                "Error: q_real CSV path must be set "
                "(Step 2 'Reference (q_real) trajectory file').",
                error=True,
            )
            return None

        control_hz = self._hz_field.model.get_value_as_float()
        measurement_length_s = self._len_field.model.get_value_as_float()
        column_prefix = self._prefix_field.model.get_value_as_string().strip()
        try:
            num_steps = expected_num_samples(control_hz, measurement_length_s)
            q_real = load_hand_trajectory_csv(
                q_real_path, column_prefix=column_prefix, num_samples=num_steps
            )
        except Exception as e:  # noqa: BLE001 - UI callback, surface on screen
            carb.log_error(f"[Parameter Tuning] q_real load failed: {e}")
            self._set_status(f"Error loading q_real CSV: {e}", error=True)
            return None

        csv_format = describe_csv_format(q_real_path, column_prefix=column_prefix)
        carb.log_info(
            f"[Parameter Tuning] q_real loaded: "
            f"{q_real.num_samples} samples x {q_real.num_joints} joints, "
            f"joints={q_real.joint_names}, csv_format={csv_format}"
        )
        self._last_q_real = q_real
        return q_real

    def _on_tuning_start_clicked(self) -> None:
        # 이미 돌고 있으면: 이 버튼은 "중단 요청"으로 동작 (현재 세대까지만 마치고 종료).
        if self._tuning_task is not None and not self._tuning_task.done():
            self._tuning_should_stop = True
            self._set_status("Stop requested - finishing current generation...")
            return

        # 두 번째 구간(버튼 클릭 → 튜닝 완료) 소요시간의 시작점. 아래 검증에서
        # 걸려 return 되면 튜닝이 안 도니 로그도 안 쓰인다(다음 클릭 때 덮어씀).
        self._t2_tuning_started_at = time.time()

        algorithm = self._algorithm_combo_model.get_selected_value()
        learning_param_joints = self._learning_params_field.model.get_value_as_int()
        population = self._population_field.model.get_value_as_int()
        generations = self._generations_field.model.get_value_as_int()
        env_count = self._env_field.model.get_value_as_int()
        loss_indicator = self._loss_combo_model.get_selected_value()
        prim_path = self._robot_combo_model.get_selected_path()
        control_hz = self._hz_field.model.get_value_as_float()
        measurement_length_s = self._len_field.model.get_value_as_float()

        if algorithm != "CMA-ES":
            self._set_status(
                f"Error: '{algorithm}' is not implemented. Only CMA-ES is available.",
                error=True,
            )
            return
        if population < 2 or generations < 1 or env_count < 1 or learning_param_joints < 1:
            self._set_status(
                "Error: Population>=2, Generations>=1, Env>=1, Learning parameters>=1 required",
                error=True,
            )
            return
        if loss_indicator not in LOSS_KINDS:
            self._set_status(f"Error: unknown loss '{loss_indicator}'", error=True)
            return
        if not prim_path:
            self._set_status("Error: select a robot in Robot Selection", error=True)
            return
        if self._last_q_ref is None:
            self._set_status(
                "Error: no q_ref trajectory yet. Run Step 1 'Data collection' first.",
                error=True,
            )
            return
        if self._load_q_real_from_field() is None:
            return

        # 옵션 A: 병렬 env = GridCloner(replicate_physics=False) + copy_from_source
        #         + 관계 기반 충돌 그룹. Env>1 이면 병렬, 아니면 단일 env.
        #         (replicate_physics=True 는 이 환경에서 하드 크래시 이력 → 안 씀.)
        effective_env = min(env_count, population)
        parallel = effective_env > 1
        if parallel:
            if effective_env != env_count:
                carb.log_warn(
                    f"[Parameter Tuning] Env={env_count} capped to {effective_env} "
                    f"(= population)."
                )
            if effective_env > 32:
                carb.log_warn(
                    f"[Parameter Tuning] Cloning {effective_env} robots (full "
                    "copies) - high RAM/VRAM use. If the process is killed, "
                    "lower Population/Env."
                )
            runner = ParallelTwinRolloutRunner(
                articulation_prim_path=prim_path,
                q_ref=self._last_q_ref,
                control_hz=control_hz,
                measurement_length_s=measurement_length_s,
                env_count=effective_env,
            )
        else:
            runner = TwinRolloutRunner(
                articulation_prim_path=prim_path,
                q_ref=self._last_q_ref,
                control_hz=control_hz,
                measurement_length_s=measurement_length_s,
            )

        # 스테이지 복제(GridCloner)는 asyncio task 밖에서 동기로 먼저 끝낸다 —
        # task 안에서 하면 다른 익스텐션 리빌드 코루틴과 충돌해
        # "Cannot enter into task ... while another task is being executed" 폭주.
        try:
            runner.build_scene()
        except Exception as e:  # noqa: BLE001
            carb.log_error(f"[Parameter Tuning] build_scene failed: {e}")
            self._set_status(f"Error: {e}", error=True)
            try:
                runner.teardown()
            except Exception:  # noqa: BLE001
                pass
            return

        self._tuning_should_stop = False
        self._tuning_progress_model.set_value(0.0)
        self._generation_label.text = f"Generation 0 / {generations}"
        self._best_loss_label.text = "-"
        _mode = "single env" if not parallel else f"{effective_env} parallel envs"
        self._set_status(
            f"CMA-ES tuning started: {population} pop x {generations} gen, {_mode}. "
            "This can take a while - watch the Isaac Sim console log."
        )

        self._tuning_task = asyncio.ensure_future(
            self._run_tuning_async(
                runner=runner,
                learning_param_joints=learning_param_joints,
                population=population,
                generations=generations,
                loss_kind=loss_indicator,
            )
        )

    async def _run_tuning_async(
        self,
        runner,
        learning_param_joints: int,
        population: int,
        generations: int,
        loss_kind: str,
    ) -> None:
        parallel = hasattr(runner, "rollout_batch")
        try:
            await runner.setup()

            num_dof = len(runner.sim_joint_names)
            # Learning parameters: 앞 L개 관절만 튜닝, 나머지는 기본 게인으로 고정.
            tuned_j = learning_param_joints
            if tuned_j > num_dof:
                raise ValueError(
                    f"Learning parameters({tuned_j}) > twin DOF count ({num_dof}). "
                    f"Set it to {num_dof} or less."
                )
            frozen = tuned_j < num_dof

            init_s_full = np.asarray(runner.default_stiffness, dtype=np.float64).ravel()
            init_d_full = np.asarray(runner.default_damping, dtype=np.float64).ravel()
            init_s = init_s_full[:tuned_j]
            init_d = init_d_full[:tuned_j]
            # self._last_q_real 은 _on_tuning_start_clicked 에서 이미 로드해둠 —
            # runner의 q_ref 재배열과 별개로, loss 비교용 정답 궤적을 트윈 DOF
            # 순서에 맞게 재배열한다.
            q_real_aligned = _reorder_to_sim_joint_order(
                self._last_q_real, runner.sim_joint_names
            )

            def _to_full(cand_s, cand_d):
                s = init_s_full.copy()
                d = init_d_full.copy()
                s[:tuned_j] = cand_s
                d[:tuned_j] = cand_d
                return s, d

            carb.log_info(
                f"[Parameter Tuning] Step2 CMA-ES: dof={num_dof}, tuned_joints={tuned_j}"
                f"{' (rest frozen)' if frozen else ''}, population={population}, "
                f"generations={generations}, "
                f"env={'single' if not parallel else runner.env_count}, loss={loss_kind}, "
                f"init_stiffness={init_s_full.tolist()}, init_damping={init_d_full.tolist()}"
            )

            config = TuningConfig(
                population=population,
                generations=generations,
                loss_kind=loss_kind,
                seed=0,
            )
            carb.log_info(
                "[Parameter Tuning] search space: "
                f"log_space={config.use_log_space} "
                f"(range x{config.log_range_factor:g}, abs_hi "
                f"Kp={config.stiffness_abs_hi:g}/Kd={config.damping_abs_hi:g}), "
                f"penalties effort_w={config.effort_weight:g}, "
                f"damping_ratio_w={config.damping_ratio_weight:g} "
                f"(zeta in [{config.damping_ratio_min:g},{config.damping_ratio_max:g}], "
                f"I={config.damping_ratio_inertia:g})"
            )

            if parallel:
                chunk_size = runner.env_count

                async def evaluate_fn(candidates):
                    losses = []
                    # 후보를 env 개수만큼씩 잘라 배치 rollout.
                    for start in range(0, len(candidates), chunk_size):
                        chunk = candidates[start : start + chunk_size]
                        sb = np.stack([_to_full(s, d)[0] for s, d in chunk])
                        db = np.stack([_to_full(s, d)[1] for s, d in chunk])
                        q_sim_batch = await runner.rollout_batch(sb, db)  # (P, N, J)
                        for k in range(len(chunk)):
                            losses.append(
                                compute_loss(q_sim_batch[k], q_real_aligned, loss_kind)
                            )
                    return losses
            else:
                async def evaluate_fn(candidates):
                    losses = []
                    for cand_s, cand_d in candidates:
                        full_s, full_d = _to_full(cand_s, cand_d)
                        q_sim = await runner.rollout(full_s, full_d)  # (num_steps, J)
                        losses.append(compute_loss(q_sim, q_real_aligned, loss_kind))
                    return losses

            def _progress(gen: int, best_loss: float, fraction: float) -> None:
                self._tuning_progress_model.set_value(fraction)
                self._generation_label.text = f"Generation {gen} / {generations}"
                self._best_loss_label.text = f"{best_loss:.6g}"
                self._set_status(
                    f"CMA-ES generation {gen}/{generations} - best {loss_kind} "
                    f"loss {best_loss:.6g}"
                )

            result = await run_cma_tuning(
                evaluate_fn=evaluate_fn,
                init_stiffness=init_s,
                init_damping=init_d,
                config=config,
                progress_cb=_progress,
                should_stop=lambda: self._tuning_should_stop,
            )

            # 튜닝한 L개 관절 결과를 전체 관절 배열로 재구성 (나머지는 기본값 유지).
            best_s_full, best_d_full = _to_full(result.best_stiffness, result.best_damping)
            self._tuned_stiffness = best_s_full
            self._tuned_damping = best_d_full
            self._tuned_joint_names = list(runner.sim_joint_names)
            self._params_table_frame.rebuild()

            # 튜닝 결과 게인을 CSV로도 저장 (USD 스테이지 외 파일 기록).
            try:
                self._save_tuned_gains_csv()
            except Exception as e:  # noqa: BLE001 - 저장 실패가 튜닝 자체를 막지 않게
                carb.log_warn(
                    f"[Parameter Tuning] could not write tuned_gains.csv: {e}"
                )

            # CMA-ES 세대별 손실 로그 저장 (수렴 그래프용).
            try:
                self._save_cma_history_csv(result)
            except Exception as e:  # noqa: BLE001
                carb.log_warn(
                    f"[Parameter Tuning] could not write cma_history.csv: {e}"
                )

            carb.log_info(
                f"[Parameter Tuning] Step2 done: best {loss_kind} loss "
                f"{result.best_loss:.6g} (traj {result.best_traj_loss:.6g} + penalty "
                f"{result.best_reg_penalty:.6g}) after {result.generations_run} "
                f"generations (stopped_early={result.stopped_early}, "
                f"cma_stop={result.cma_stop_reason or 'none'}). "
                f"stiffness={best_s_full.tolist()}, damping={best_d_full.tolist()}"
            )
            if result.stopped_early:
                note = " (stopped by user)"
            elif result.cma_stop_reason:
                note = f" (CMA-ES converged: {result.cma_stop_reason})"
            else:
                note = ""
            self._set_status(
                f"Tuning done{note}: best {loss_kind} loss {result.best_loss:.6g} "
                f"after {result.generations_run} generations. "
                "Review 'Optimization parameters', then 'Apply to twin' "
                "(re-runs q_ref -> q_sim_a and lets you compare with 'Matching results')."
            )

            # 두 번째 구간 소요시간 기록: 'Tuning start' 클릭 → 여기(튜닝 완료).
            if self._t2_tuning_started_at is not None:
                self._append_timing_log(
                    "tuning_start_to_done",
                    time.time() - self._t2_tuning_started_at,
                    f"{population} pop x {generations} gen | {loss_kind} "
                    f"| best loss {result.best_loss:.6g} "
                    f"| {result.generations_run} generations run"
                    + (" | stopped early" if result.stopped_early else ""),
                )

        except asyncio.CancelledError:
            self._set_status("Tuning cancelled.", error=True)
            raise
        except Exception as e:  # noqa: BLE001 - UI callback, surface on screen
            carb.log_error(f"[Parameter Tuning] Step2 failed: {e}")
            self._set_status(f"Error: {e}", error=True)
        finally:
            # teardown(sim.stop + 스코프 삭제)은 대량 USD 변경 → 이 task 안에서 하면
            # property 창 리빌드 코루틴 폭주("Cannot enter into task"). 다음 프레임으로 미룸.
            self._active_runner = runner
            _defer_call(self._teardown_active_runner)
            self._tuning_task = None
            self._tuning_should_stop = False
            # 성공 경로에서 이미 기록했으면 None, 취소/에러로 못 왔으면 여기서 정리.
            self._t2_tuning_started_at = None

    def _teardown_active_runner(self) -> None:
        runner = self._active_runner
        self._active_runner = None
        if runner is not None:
            runner.teardown()

    def _save_tuned_gains_csv(self) -> None:
        """튜닝된 best stiffness/damping 을 tuned_gains.csv 로 저장한다.

        값은 'Optimization parameters' 표 / 관절 Property 패널과 동일한
        USD DriveAPI 단위(per degree)로 쓴다. 폴더가 없으면 만든다.
        매 실행마다 같은 파일명으로 덮어쓴다.
        """
        if self._tuned_stiffness is None or self._tuned_joint_names is None:
            return
        path = self._default_tuned_gains_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["joint", "stiffness_per_deg", "damping_per_deg"])
            for i, name in enumerate(self._tuned_joint_names):
                writer.writerow(
                    [
                        name,
                        f"{self._tuned_stiffness[i] * _RAD_TO_DEG:.6g}",
                        f"{self._tuned_damping[i] * _RAD_TO_DEG:.6g}",
                    ]
                )
        carb.log_info(f"[Parameter Tuning] tuned gains saved to {path}")

    def _save_cma_history_csv(self, result) -> None:
        """CMA-ES 세대별 손실 로그를 cma_history.csv 로 저장한다.

        컬럼: generation(1부터), best_loss(그 세대까지 best-so-far 총합),
              best_traj_loss / best_reg_penalty(best 후보의 궤적 loss / 물리 페널티,
              둘을 더하면 best_loss), gen_min_loss(그 세대 후보 최소 총합),
              gen_median_loss(그 세대 후보 총합 중앙값),
              sigma(세대 종료 시 CMA-ES step-size).
        세대 vs loss 수렴 그래프(보통 y축 로그 스케일)를 그대로 그릴 수 있다.
        폴더가 없으면 만들고, 매 실행마다 같은 파일명으로 덮어쓴다.
        """
        hist = list(getattr(result, "loss_history", []) or [])
        if not hist:
            return
        traj = list(getattr(result, "traj_loss_history", []) or [])
        reg = list(getattr(result, "reg_penalty_history", []) or [])
        gen_min = list(getattr(result, "gen_min_history", []) or [])
        gen_med = list(getattr(result, "gen_median_history", []) or [])
        sigma = list(getattr(result, "sigma_history", []) or [])
        path = self._default_cma_history_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def _cell(seq, i):
            return f"{seq[i]:.8g}" if i < len(seq) else ""

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "generation", "best_loss", "best_traj_loss", "best_reg_penalty",
                    "gen_min_loss", "gen_median_loss", "sigma",
                ]
            )
            for i, best in enumerate(hist):
                writer.writerow(
                    [
                        i + 1,
                        f"{best:.8g}",
                        _cell(traj, i),
                        _cell(reg, i),
                        _cell(gen_min, i),
                        _cell(gen_med, i),
                        _cell(sigma, i),
                    ]
                )
        carb.log_info(
            f"[Parameter Tuning] CMA-ES history ({len(hist)} generations) saved to {path}"
        )

    # ------------------------------------------------------------------
    # Step 3: Verification
    # ------------------------------------------------------------------

    def _build_params_table(self) -> None:
        # 튜닝이 끝나면 best stiffness/damping 으로 채운다. 아직이면 관절 이름만.
        # 값은 USD DriveAPI 단위(각도 드라이브 = per degree)로 환산해서 표시 →
        # "Apply to twin" 후 관절 Property 패널의 숫자와 일치한다.
        with ui.VStack(height=0, spacing=2):
            if self._tuned_stiffness is not None and self._tuned_joint_names is not None:
                ui.Label(
                    "USD drive units (per degree) - matches each joint's Property panel",
                    style={"color": 0xFF7A7A7A, "font_size": 10},
                    word_wrap=True,
                )
                rows = [
                    (
                        name,
                        f"{self._tuned_stiffness[i] * _RAD_TO_DEG:.6g}",
                        f"{self._tuned_damping[i] * _RAD_TO_DEG:.6g}",
                    )
                    for i, name in enumerate(self._tuned_joint_names)
                ]
            else:
                if self._last_q_sim is not None:
                    joint_names = self._last_q_sim.joint_names
                else:
                    joint_count = self._learning_params_field.model.get_value_as_int()
                    joint_names = [f"Joint {i + 1}" for i in range(max(joint_count, 0))]
                rows = [(name, "-", "-") for name in joint_names]
            _build_simple_table(["Joint", "Stiffness", "Damping"], rows)

    def _build_joint_error_table(self) -> None:
        if self._last_abs_error is None or self._last_q_sim is None:
            ui.Label(
                "Click Matching results first (run Step 1 data collection).",
                style={"color": 0xFF7A7A7A},
                word_wrap=True,
            )
            return

        mean_per_joint = np.mean(self._last_abs_error, axis=0)
        max_per_joint = np.max(self._last_abs_error, axis=0)
        rows = [
            (name, f"{mean_per_joint[i]:.4f}", f"{max_per_joint[i]:.4f}")
            for i, name in enumerate(self._last_q_sim.joint_names)
        ]
        _build_simple_table(["Joint", "Mean err (rad)", "Max err (rad)"], rows)

    def _build_joint_overlay_plot(self) -> None:
        if self._last_aligned_q_real is None or self._last_q_sim is None:
            ui.Label(
                "Click Matching results first.",
                style={"color": 0xFF7A7A7A},
                word_wrap=True,
            )
            return

        joint_idx = self._verification_joint_combo_model.get_selected_index()
        num_joints = self._last_aligned_q_real.shape[1]
        if not (0 <= joint_idx < num_joints):
            joint_idx = 0

        q_real_curve = self._last_aligned_q_real[:, joint_idx].tolist()
        q_sim_curve = self._last_q_sim.positions[:, joint_idx].tolist()

        # before(튜닝 전 게인) q_sim — 있으면 흐린 주황으로 같이 그린다.
        base_curve = None
        if (
            self._baseline_q_sim is not None
            and joint_idx < self._baseline_q_sim.positions.shape[1]
        ):
            base_curve = self._baseline_q_sim.positions[:, joint_idx].tolist()

        all_vals = q_real_curve + q_sim_curve + (base_curve or [])
        lo = min(all_vals)
        hi = max(all_vals)
        if hi - lo < 1e-6:
            hi = lo + 1e-6  # scale_min == scale_max면 Plot이 제대로 안 그려짐 방지

        with ui.VStack(spacing=2, height=0):
            # 같은 scale_min/scale_max로 ZStack에 겹쳐 그림 — 뒤 Plot의 background_color는
            # 완전 투명(0x00000000)으로 둬서 앞 Plot이 가려지지 않게 한다.
            with ui.ZStack(height=110):
                ui.Plot(
                    ui.Type.LINE,
                    lo,
                    hi,
                    *q_real_curve,
                    height=110,
                    style={"color": 0xFFBFBFBF, "background_color": 0xFF1E1E1E},
                )
                if base_curve is not None:
                    ui.Plot(
                        ui.Type.LINE,
                        lo,
                        hi,
                        *base_curve,
                        height=110,
                        style={"color": 0xFF5AA0E0, "background_color": 0x00000000},
                    )
                ui.Plot(
                    ui.Type.LINE,
                    lo,
                    hi,
                    *q_sim_curve,
                    height=110,
                    style={"color": 0xFF7FD8A0, "background_color": 0x00000000},
                )
            with ui.HStack(height=16):
                ui.Label("q_real", style={"color": 0xFFBFBFBF, "font_size": 10})
                if base_curve is not None:
                    ui.Label(
                        "q_sim before", style={"color": 0xFF5AA0E0, "font_size": 10}
                    )
                ui.Label(
                    "q_sim after" if base_curve is not None else "q_sim (twin)",
                    style={"color": 0xFF7FD8A0, "font_size": 10},
                )

    def _build_error_over_time_plot(self) -> None:
        if self._last_rms_error_over_time is None:
            ui.Label(
                "Click Matching results first.",
                style={"color": 0xFF7A7A7A},
                word_wrap=True,
            )
            return

        values = self._last_rms_error_over_time.tolist()
        hi = max(max(values), 1e-6)
        ui.Plot(
            ui.Type.LINE,
            0.0,
            hi,
            *values,
            height=90,
            style={"color": 0xFF7FD8A0, "background_color": 0xFF1E1E1E},
        )

    def _on_verification_joint_changed(self) -> None:
        if self._last_aligned_q_real is not None:
            self._joint_overlay_plot_frame.rebuild()

    def _compute_match_metrics(self, q_real, q_sim) -> dict:
        """q_real vs q_sim 비교 지표. _reorder_to_sim_joint_order 로 관절 정렬."""
        aligned = _reorder_to_sim_joint_order(q_real, q_sim.joint_names)
        diff = aligned - q_sim.positions
        abs_err = np.abs(diff)
        mean_error = float(np.mean(abs_err))
        return {
            "aligned_q_real": aligned,
            "abs_error": abs_err,
            "rms_over_time": np.sqrt(np.mean(diff**2, axis=1)),
            "mean_error": mean_error,
            "max_error": float(np.max(abs_err)),
            # 휴리스틱: mean_error 0 → 100%, _MATCH_SCORE_TOLERANCE_RAD 이상 → 0%.
            "match_score": max(0.0, 1.0 - mean_error / _MATCH_SCORE_TOLERANCE_RAD) * 100.0,
        }

    def _apply_match_metrics_to_ui(self, m: dict) -> None:
        self._last_aligned_q_real = m["aligned_q_real"]
        self._last_abs_error = m["abs_error"]
        self._last_rms_error_over_time = m["rms_over_time"]

        b = self._baseline_metrics

        def _pair(cur: float, was, pct: bool = False) -> str:
            s = f"{cur:.1f}%" if pct else f"{cur:.3f} rad"
            if was is not None:
                s += f"  (was {was:.1f}%)" if pct else f"  (was {was:.3f})"
            return s

        self._match_score_label.text = _pair(
            m["match_score"], b["match_score"] if b else None, pct=True
        )
        self._mean_error_label.text = _pair(m["mean_error"], b["mean_error"] if b else None)
        self._max_error_label.text = _pair(m["max_error"], b["max_error"] if b else None)

        self._joint_error_table_frame.rebuild()
        self._joint_overlay_plot_frame.rebuild()
        self._error_over_time_plot_frame.rebuild()

    def _improvement_text(self, m: dict) -> str:
        b = self._baseline_metrics
        if not b:
            return (
                f"mean error {m['mean_error']:.3f} rad, match {m['match_score']:.1f}%"
            )
        de = b["mean_error"] - m["mean_error"]
        pct = (de / b["mean_error"] * 100.0) if b["mean_error"] > 0 else 0.0
        return (
            f"mean error {m['mean_error']:.3f} rad (was {b['mean_error']:.3f}) "
            f"-> {pct:+.1f}% ; match {m['match_score']:.1f}% (was {b['match_score']:.1f}%)"
        )

    def _on_apply_to_twin_clicked(self) -> None:
        if self._tuned_stiffness is None or self._tuned_joint_names is None:
            self._set_status(
                "Error: no tuned parameters yet. Run Step 2 'Tuning start' first.",
                error=True,
            )
            return
        if self._apply_task is not None and not self._apply_task.done():
            self._set_status("Apply already in progress...")
            return
        if self._tuning_task is not None and not self._tuning_task.done():
            self._set_status("Error: wait for tuning to finish first.", error=True)
            return
        prim_path = self._robot_combo_model.get_selected_path()
        control_hz = self._hz_field.model.get_value_as_float()
        measurement_length_s = self._len_field.model.get_value_as_float()
        show_in_viewport = self._show_viewport_cb.model.get_value_as_bool()
        self._set_status("Applying tuned gains to twin (saving to USD)...")
        self._apply_task = asyncio.ensure_future(
            self._apply_to_twin_async(
                prim_path, control_hz, measurement_length_s, show_in_viewport
            )
        )

    async def _apply_to_twin_async(
        self,
        prim_path: str,
        control_hz: float,
        measurement_length_s: float,
        show_in_viewport: bool,
    ) -> None:
        # Apply to twin: 튜닝된 게인을 USD에 저장하고, 곧바로 원래 q_ref(Step 1에서
        # 로드했던 그 궤적, self._last_q_ref)를 새 게인으로 다시 재생해서
        # "적용 후" 궤적을 얻어 q_sim_a로 저장한다 — 예전엔 이 재생을 "Data
        # collection"을 따로 눌러야만 했는데(Apply와 검증을 분리하려던 의도였음),
        # 실제로 써보니 재수집을 깜빡하고 Matching results를 누르면 튜닝 전 q_sim
        # 그대로 비교돼 혼란스럽다는 피드백이 있어 다시 합쳤다.
        #
        # "튜닝 전" 스냅샷(baseline)은 게인을 덮어쓰기 직전에 찍는다 — 그 시점의
        # self._last_q_real/self._last_q_sim이 곧 "before" 상태이고, 이후
        # Matching results를 눌렀을 때 "(was 72.3%)" 비교 문구와 오버레이 그래프의
        # "q_sim before"(파란 선)가 이 값을 쓴다. 그 다음 self._last_q_sim을 새로
        # 재생한 q_sim_a로 덮어써서, Matching results가 별다른 수정 없이도 바로
        # "실제 vs 적용 후"를 "현재"로, "실제 vs 적용 전"을 baseline("was")으로
        # 비교하게 만든다.
        gains_saved = False
        try:
            if self._last_q_real is not None and self._last_q_sim is not None:
                try:
                    self._baseline_metrics = self._compute_match_metrics(
                        self._last_q_real, self._last_q_sim
                    )
                    self._baseline_q_sim = self._last_q_sim
                except ValueError:
                    self._baseline_metrics = None
                    self._baseline_q_sim = None

            await author_drive_gains_to_usd(
                articulation_prim_path=prim_path,
                stiffness=self._tuned_stiffness,
                damping=self._tuned_damping,
                control_hz=control_hz,
            )
            gains_saved = True

            # USD 스테이지에 쓰는 것과 함께 tuned_gains.csv 로도 남긴다.
            try:
                self._save_tuned_gains_csv()
            except Exception as e:  # noqa: BLE001
                carb.log_warn(
                    f"[Parameter Tuning] could not write tuned_gains.csv: {e}"
                )

            if self._last_q_ref is None:
                self._set_status(
                    f"Applied tuned Kp/damping to {prim_path} (saved to USD), but no "
                    "q_ref in memory to re-run - run 'Data collection' once first, "
                    "then 'Apply to twin' again (or 'Matching results' manually).",
                )
                return

            q_sim_after = await run_trajectory_on_twin(
                articulation_prim_path=prim_path,
                q_ref=self._last_q_ref,
                control_hz=control_hz,
                measurement_length_s=measurement_length_s,
                show_in_viewport=show_in_viewport,
            )
            save_trajectory_csv(
                self._default_output_path_after_apply,
                q_sim_after.joint_names,
                q_sim_after.timestamps,
                q_sim_after.positions,
            )

            # "현재" q_sim을 적용 후 궤적으로 교체 — Matching results가 이걸 바로
            # "실제 vs 적용 후"로 쓴다. 이전 매칭 결과(오차/그래프)는 이 새 q_sim
            # 기준으로 다시 계산해야 하므로 캐시를 비운다.
            self._last_q_sim = q_sim_after
            self._last_aligned_q_real = None
            self._last_abs_error = None
            self._last_rms_error_over_time = None
            self._verification_joint_combo_model.replace_item_list(q_sim_after.joint_names)
            self._joint_error_table_frame.rebuild()
            self._joint_overlay_plot_frame.rebuild()
            self._error_over_time_plot_frame.rebuild()

            self._set_status(
                f"Applied tuned Kp/damping to {prim_path} (saved to USD) and re-ran "
                f"q_ref on the twin -> q_sim_a ({q_sim_after.num_samples} samples) "
                f"saved to {self._default_output_path_after_apply}. "
                "Click 'Matching results' to compare before vs after."
            )
        except Exception as e:  # noqa: BLE001 - UI callback, surface on screen
            carb.log_error(f"[Parameter Tuning] Apply to twin failed: {e}")
            if gains_saved:
                self._set_status(
                    f"Gains applied to {prim_path} (saved to USD), but re-running "
                    f"q_ref to build q_sim_a failed: {e}",
                    error=True,
                )
            else:
                self._set_status(f"Error: {e}", error=True)
        finally:
            self._apply_task = None

    def _on_matching_results_clicked(self) -> None:
        if self._last_q_sim is None:
            self._set_status(
                "Error: no q_sim to compare yet. Run Step 1 'Data collection' first.",
                error=True,
            )
            return
        if self._load_q_real_from_field() is None:
            return

        try:
            m = self._compute_match_metrics(self._last_q_real, self._last_q_sim)
        except ValueError as e:
            self._set_status(f"Error: {e}", error=True)
            return

        self._apply_match_metrics_to_ui(m)

        self._set_status(f"Matching results: {self._improvement_text(m)}")

        # 첫 구간 소요시간 기록: q_ref 파일 선택 → 여기('Matching results' 완료).
        # 한 번 기록하면 시작 시각을 비워, 다시 재려면 레퍼런스 파일을 다시 골라야 한다.
        if self._t1_ref_selected_at is not None:
            q_ref_path = self._csv_path_field.model.get_value_as_string().strip()
            self._append_timing_log(
                "ref_select_to_matching_results",
                time.time() - self._t1_ref_selected_at,
                f"q_ref={q_ref_path} | match {m['match_score']:.1f}% "
                f"| mean err {m['mean_error']:.4f} rad",
            )
            self._t1_ref_selected_at = None

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status_label.text = text
        self._status_label.style = {"color": 0xFF5C5CFF if error else 0xFF7FD8A0}

    def on_shutdown(self) -> None:
        carb.log_info("[Parameter Tuning] on_shutdown")
        if self._data_collection_task is not None and not self._data_collection_task.done():
            self._data_collection_task.cancel()
            self._data_collection_task = None
        self._tuning_should_stop = True
        if self._tuning_task is not None and not self._tuning_task.done():
            self._tuning_task.cancel()
            self._tuning_task = None
        if self._apply_task is not None and not self._apply_task.done():
            self._apply_task.cancel()
            self._apply_task = None
        # 미뤄둔 teardown 이 아직 안 돌았으면 여기서 즉시 (익스텐션 언로드 중이라 폭주 무관)
        if self._active_runner is not None:
            try:
                self._teardown_active_runner()
            except Exception:  # noqa: BLE001
                pass
        if self._file_picker:
            self._file_picker.destroy()
            self._file_picker = None
        if self._window:
            self._window.destroy()
            self._window = None
