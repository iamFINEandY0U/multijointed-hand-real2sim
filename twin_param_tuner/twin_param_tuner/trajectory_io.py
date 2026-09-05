"""
Mode B - 궤적 CSV 입출력.

Isaac Sim 런타임에 의존하지 않는 순수 Python 모듈. q_ref/q_real(입력)과
q_sim(출력) CSV를 동일한 포맷으로 읽고 쓴다.

CSV 포맷 (헤더 필수):
    time, <joint_name_1>, <joint_name_2>, ..., <joint_name_N>
    0.000, 0.0123, -0.0210, ...
    0.003, 0.0125, -0.0200, ...
    ...

- `time` 컬럼은 선택 사항. 없으면 이후 재생 단계에서 제어 주기로부터
  타임스탬프를 다시 생성한다.
- 관절 컬럼 순서는 트윈 Articulation의 DOF 순서와 반드시 일치해야 한다.
  (twin_playback.run_trajectory_on_twin 이 실행 전에 검증한다.)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

# Delto DG-3F 궤적 CSV 의 관절 컬럼 패턴: 'f1_j1_deg', 'f3_j4_deg' 등
# (finger N, joint M). 트윈 USD 조인트 이름 'j_N_M' 로 정규화하는 데 쓴다.
_DG3F_JOINT_RE = re.compile(r"^f(\d+)_j(\d+)")
_TIME_COLUMN_NAMES = ("t_s", "t", "time", "timestamp")


@dataclass
class Trajectory:
    joint_names: List[str]
    timestamps: np.ndarray   # shape (N,)  seconds
    positions: np.ndarray    # shape (N, num_joints) radians

    def __post_init__(self) -> None:
        if self.positions.ndim != 2:
            raise ValueError(f"positions must be 2D (N, num_joints), got shape {self.positions.shape}")
        if self.positions.shape[1] != len(self.joint_names):
            raise ValueError(
                f"positions has {self.positions.shape[1]} joint columns but "
                f"{len(self.joint_names)} joint_names were given"
            )
        if self.positions.shape[0] != self.timestamps.shape[0]:
            raise ValueError(
                f"positions has {self.positions.shape[0]} rows but "
                f"timestamps has {self.timestamps.shape[0]} entries"
            )

    @property
    def num_samples(self) -> int:
        return self.positions.shape[0]

    @property
    def num_joints(self) -> int:
        return self.positions.shape[1]


def load_trajectory_csv(path: Union[str, Path]) -> Trajectory:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV is empty: {path}")
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"CSV has a header row but no data rows: {path}")

    has_time_col = header[0].strip().lower() in ("time", "t", "timestamp")
    joint_names = [h.strip() for h in (header[1:] if has_time_col else header)]

    try:
        data = np.array(rows, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Failed to parse CSV values (non-numeric cell?): {path}") from e

    if data.ndim != 2:
        raise ValueError(f"CSV rows have inconsistent length: {path}")

    if has_time_col:
        timestamps = data[:, 0]
        positions = data[:, 1:]
    else:
        # 시간 컬럼이 없으면 일단 스텝 인덱스로 채워두고,
        # run_trajectory_on_twin 호출 시 control_hz 기준으로 재계산한다.
        timestamps = np.arange(len(rows), dtype=np.float64)
        positions = data

    if positions.shape[1] != len(joint_names):
        raise ValueError(
            f"Column count mismatch ({path}): {positions.shape[1]} value columns, "
            f"{len(joint_names)} joint names"
        )

    return Trajectory(joint_names=joint_names, timestamps=timestamps, positions=positions)


def save_trajectory_csv(
    path: Union[str, Path],
    joint_names: List[str],
    timestamps: np.ndarray,
    positions: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if positions.shape[0] != timestamps.shape[0]:
        raise ValueError("timestamps and positions have different row counts")
    if positions.shape[1] != len(joint_names):
        raise ValueError("joint_names count differs from positions column count")

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *joint_names])
        for t, row in zip(timestamps, positions):
            writer.writerow([f"{t:.6f}", *[f"{v:.6f}" for v in row]])


def save_comparison_csv(
    path: Union[str, Path],
    joint_names: List[str],
    timestamps: np.ndarray,
    q_real: np.ndarray,
    q_sim_before: np.ndarray,
    q_sim_after: np.ndarray,
) -> None:
    """q_real / q_sim(튜닝 전) / q_sim(튜닝 후) 를 관절별로 나란히 붙여 CSV 1장으로 저장.

    세 배열 모두 shape (N, num_joints) — timestamps 길이 N, joint_names 순서와
    일치해야 한다(호출자가 미리 같은 관절 순서로 정렬해서 넘겨야 함. 예:
    extension.py 는 `_compute_match_metrics`의 aligned_q_real / baseline q_sim /
    최신 q_sim 을 그대로 넘긴다 — 셋 다 트윈 DOF 순서로 이미 맞춰져 있음).

    출력 헤더: time, <joint>_q_real, <joint>_q_sim_before, <joint>_q_sim_after, ...
    (관절마다 3개 컬럼씩 반복 — 관절별로 세 값을 나란히 비교하기 쉽게 배치).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(joint_names)
    for label, arr in (
        ("q_real", q_real),
        ("q_sim_before", q_sim_before),
        ("q_sim_after", q_sim_after),
    ):
        if arr.shape != (timestamps.shape[0], n):
            raise ValueError(
                f"{label} shape {arr.shape} differs from (timestamps {timestamps.shape[0]}, "
                f"joints {n})"
            )

    header = ["time"]
    for jn in joint_names:
        header += [f"{jn}_q_real", f"{jn}_q_sim_before", f"{jn}_q_sim_after"]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(timestamps):
            row = [f"{t:.6f}"]
            for j in range(n):
                row += [
                    f"{q_real[i, j]:.6f}",
                    f"{q_sim_before[i, j]:.6f}",
                    f"{q_sim_after[i, j]:.6f}",
                ]
            writer.writerow(row)


def load_mujoco_hand_csv(
    path: Union[str, Path],
    column_prefix: str = "qpos_",
    num_samples: "Union[int, None]" = None,
    time_column: str = "t",
) -> Trajectory:
    """
    MuJoCo 스타일로 기록된 Allegro Hand 궤적 CSV 전용 로더.

    실제 파일 헤더 예시:
        t, qpos_ffj0, qpos_ffj1, ..., qpos_thj3,
        qvel_ffj0, ..., ctrl_ffj0, ..., tau_ffj0, ..., ff_tip_x, ...

    이 함수는 `time_column`과, `column_prefix`로 시작하는 컬럼들만 뽑아서
    Trajectory로 반환한다. 나머지(qvel/ctrl/tau/tip 등)는 무시한다.

    Args:
        path: CSV 경로
        column_prefix: 뽑을 컬럼 접두어. 기본 "qpos_" (실제 도달 위치).
            "ctrl_" 등 다른 접두어로도 재사용 가능.
        num_samples: 앞에서부터 몇 개 행만 쓸지. None이면 전체 사용.
            (예: 333Hz x 3.0s 만 쓰려면 999)
        time_column: 시간 컬럼 이름. 기본 "t".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [row for row in reader if row]

    if time_column not in header:
        raise ValueError(f"Time column '{time_column}' not found in header: {header[:5]}...")

    target_cols = [i for i, h in enumerate(header) if h.startswith(column_prefix)]
    if not target_cols:
        raise ValueError(f"No column starts with '{column_prefix}'. Header: {header}")

    time_idx = header.index(time_column)
    joint_names = [header[i][len(column_prefix):] for i in target_cols]

    if num_samples is not None:
        if num_samples > len(rows):
            raise ValueError(
                f"num_samples({num_samples}) exceeds CSV row count ({len(rows)})"
            )
        rows = rows[:num_samples]

    try:
        data = np.array(rows, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Failed to parse CSV values: {path}") from e

    timestamps = data[:, time_idx]
    positions = data[:, target_cols]

    return Trajectory(joint_names=joint_names, timestamps=timestamps, positions=positions)


def load_dg3f_hand_csv(
    path: Union[str, Path],
    num_samples: "Union[int, None]" = None,
) -> Trajectory:
    """Delto DG-3F 그리퍼 궤적 CSV 전용 로더.

    실제 파일 헤더 예시:
        t_s, f1_j1_deg, f1_j2_deg, f1_j3_deg, f1_j4_deg,
             f2_j1_deg, ..., f3_j4_deg, ramp

    - 시간 컬럼: `t_s` / `t` / `time` / `timestamp` 중 아무거나. 있으면 값만 읽고
      (재생 단계에서 control_hz 로 다시 만든다), 없으면 스텝 인덱스로 채운다.
    - 관절 컬럼: 'fN_jM...' 로 시작하는 컬럼만 사용. 'ramp' 등 나머지는 무시한다.
    - 값 단위는 **degree** 로 간주하고 **radian 으로 변환**한다
      (Isaac 물리 텐서 API 는 회전 관절을 rad 로 다룬다).
    - 관절 이름은 트윈(delto_t.usd 등 URDF 임포트) DOF 이름과 같은
      'F{N}M{M}' 형태('F1M1' … 'F3M4')로 반환한다 → twin_playback 이름
      매칭이 변환 없이 그대로 통한다. ('j_1_1' 이름을 쓰는 dg3f.usd 등
      다른 에셋도 twin_playback 에서 역변환으로 계속 매칭된다.)

    Args:
        path: CSV 경로
        num_samples: 앞에서부터 몇 개 행만 쓸지. None 이면 전체 사용.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"CSV is empty: {path}")
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"CSV has a header row but no data rows: {path}")

    joint_cols = [i for i, h in enumerate(header) if _DG3F_JOINT_RE.match(h)]
    if not joint_cols:
        raise ValueError(
            f"DG-3F joint columns ('fN_jM...') not found in header: {header}"
        )

    def _canon(h: str) -> str:
        m = _DG3F_JOINT_RE.match(h)
        # 트윈(delto_t.usd) DOF 이름과 동일한 'F{finger}M{motor}' 로 뽑는다 →
        # q_sim / Property 패널 / tuned_gains.csv 와 관절 이름이 끝까지 일치.
        return f"F{m.group(1)}M{m.group(2)}"

    joint_names = [_canon(header[i]) for i in joint_cols]

    time_idx = next(
        (i for i, h in enumerate(header) if h.lower() in _TIME_COLUMN_NAMES), None
    )

    if num_samples is not None:
        if num_samples > len(rows):
            raise ValueError(
                f"num_samples({num_samples}) exceeds CSV row count ({len(rows)})"
            )
        rows = rows[:num_samples]

    try:
        data = np.array(rows, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Failed to parse CSV values: {path}") from e

    positions = np.deg2rad(data[:, joint_cols])
    if time_idx is not None:
        timestamps = data[:, time_idx]
    else:
        timestamps = np.arange(len(rows), dtype=np.float64)

    return Trajectory(joint_names=joint_names, timestamps=timestamps, positions=positions)


def _classify_csv_header(header: List[str], column_prefix: str) -> str:
    """헤더만 보고 load_hand_trajectory_csv 가 고를 로더 종류를 판별.

    반환값: "dg3f" | "mujoco" | "generic". 판별 기준은 load_hand_trajectory_csv
    의 docstring 참고. describe_csv_format()과 공유하는 판별 로직.
    """
    if any(_DG3F_JOINT_RE.match(h) for h in header):
        return "dg3f"
    lower = [h.lower() for h in header]
    if any(h.startswith(column_prefix) for h in header) and "t" in lower:
        return "mujoco"
    return "generic"


_FORMAT_LABELS = {
    "dg3f": "detected as DG-3F format (prefix ignored)",
    "mujoco": "detected as MuJoCo/Allegro format (prefix applied)",
    "generic": "detected as generic CSV format (prefix ignored)",
}


def detect_csv_kind(path: Union[str, Path], column_prefix: str = "qpos_") -> str:
    """load_hand_trajectory_csv 가 이 CSV에 고를 로더 종류를 반환.

    반환값: "dg3f" | "mujoco" | "generic". 헤더 한 줄만 읽고 전체 파싱은 하지
    않는다 — UI가 "prefix가 실제로 쓰이는 경우("mujoco")에만 CSV column prefix
    입력칸을 보여줄지" 같은 판단에 쓴다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    with path.open("r", newline="") as f:
        try:
            header = [h.strip() for h in next(csv.reader(f))]
        except StopIteration:
            raise ValueError(f"CSV is empty: {path}")
    return _classify_csv_header(header, column_prefix)


def describe_csv_format(path: Union[str, Path], column_prefix: str = "qpos_") -> str:
    """load_hand_trajectory_csv 가 이 CSV를 어떤 포맷으로 인식할지 사람이 읽을 수
    있는 문자열로 반환 (상태줄 표시용)."""
    return _FORMAT_LABELS[detect_csv_kind(path, column_prefix)]


def load_hand_trajectory_csv(
    path: Union[str, Path],
    column_prefix: str = "qpos_",
    num_samples: "Union[int, None]" = None,
) -> Trajectory:
    """헤더를 보고 알맞은 로더를 자동 선택한다.

      1) 'fN_jM...' 컬럼이 있으면 → DG-3F 로더 (degree, 'j_N_M' 이름).
      2) `column_prefix` 로 시작하는 컬럼 + 't' 시간 컬럼이 있으면 → MuJoCo
         Allegro 로더 (rad, prefix 뗀 이름).
      3) 그 외 → 일반 로더 (time 열 optional, 값은 rad 로 간주).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")

    with path.open("r", newline="") as f:
        try:
            header = [h.strip() for h in next(csv.reader(f))]
        except StopIteration:
            raise ValueError(f"CSV is empty: {path}")

    kind = _classify_csv_header(header, column_prefix)

    if kind == "dg3f":
        return load_dg3f_hand_csv(path, num_samples=num_samples)

    if kind == "mujoco":
        return load_mujoco_hand_csv(
            path, column_prefix=column_prefix, num_samples=num_samples
        )

    traj = load_trajectory_csv(path)
    if num_samples is not None:
        if num_samples > traj.num_samples:
            raise ValueError(
                f"num_samples({num_samples}) exceeds CSV row count ({traj.num_samples})"
            )
        traj = Trajectory(
            joint_names=traj.joint_names,
            timestamps=traj.timestamps[:num_samples],
            positions=traj.positions[:num_samples],
        )
    return traj


def expected_num_samples(control_hz: float, measurement_length_s: float) -> int:
    """예: 333 Hz x 3.0 s = 999 samples."""
    return round(control_hz * measurement_length_s)


def validate_against_control_params(
    traj: Trajectory, control_hz: float, measurement_length_s: float
) -> None:
    """궤적(q_ref 또는 q_real) 샘플 수가 (제어 주기 x 측정 길이)와 맞는지 검증.

    안 맞으면 재생 시 무슨 값을 어떤 스텝에 넣을지 애매해지므로,
    여기서 명확한 에러로 막는다 (임의로 리샘플링/보간하지 않음).
    """
    expected = expected_num_samples(control_hz, measurement_length_s)
    if traj.num_samples != expected:
        raise ValueError(
            f"Trajectory sample count ({traj.num_samples}) != "
            f"control_hz({control_hz}) x measurement_length_s({measurement_length_s}) "
            f"= {expected}. Re-export the trajectory file or fix the input values."
        )
