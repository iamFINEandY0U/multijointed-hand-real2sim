"""Sandbox tests for twin_param_tuner.twin_playback 의 관절 이름 매칭 로직.

실행: `python3 tests/test_twin_playback.py` (numpy 필요).

extension.py 는 carb/omni 를 import 하므로 패키지 __init__ 을 거치지 않고,
twin_playback.py 만 로드한다. 단 이 모듈은 `from .trajectory_io import ...`
(상대 import) 를 쓰므로, 가짜 패키지를 먼저 등록해 준다.
"""
import importlib
import importlib.util
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_DIR = os.path.join(_ROOT, "twin_param_tuner")

_pkg = types.ModuleType("twin_param_tuner")
_pkg.__path__ = [_PKG_DIR]
sys.modules["twin_param_tuner"] = _pkg
importlib.import_module("twin_param_tuner.trajectory_io")

_spec = importlib.util.spec_from_file_location(
    "twin_param_tuner.twin_playback", os.path.join(_PKG_DIR, "twin_playback.py")
)
twin_playback = importlib.util.module_from_spec(_spec)
sys.modules["twin_param_tuner.twin_playback"] = twin_playback
_spec.loader.exec_module(twin_playback)

_map_csv_names_to_sim = twin_playback._map_csv_names_to_sim

# DG-3F CSV 로더가 뽑는 관절 이름 (이제 delto_t.usd 와 같은 'F{N}M{M}').
_DG3F_CSV = [f"F{f}M{j}" for f in (1, 2, 3) for j in (1, 2, 3, 4)]
# dg3f.usd(isaacsim 패키지) 정규형 — 로더가 역호환으로 계속 받아들여야 하는 형태.
_DG3F_CANON = [f"j_{f}_{j}" for f in (1, 2, 3) for j in (1, 2, 3, 4)]


def test_dg3f_csv_matches_delto_t_usd():
    # delto_t.usd (URDF 임포트): 트윈 DOF 이름이 CSV 이름과 그대로 같음.
    assert _map_csv_names_to_sim(_DG3F_CSV, list(_DG3F_CSV)) == _DG3F_CSV


def test_dg3f_csv_matches_isaacsim_asset():
    # dg3f.usd (isaacsim): 'j_1_1' … — 'F1M1' CSV 이름이 역변환으로 매칭돼야 함.
    assert _map_csv_names_to_sim(_DG3F_CSV, list(_DG3F_CANON)) == _DG3F_CANON


def test_dg3f_canon_matches_isaacsim_asset():
    # dg3f.usd (isaacsim): 트윈 DOF 이름이 CSV 이름과 그대로 같음.
    assert _map_csv_names_to_sim(_DG3F_CANON, list(_DG3F_CANON)) == _DG3F_CANON


def test_dg3f_canon_matches_urdf_import_fnmn():
    # delto.usd / delto_gripper_3f.usd (URDF 임포트): 'F1M1' … 'F3M4'.
    sim = [f"F{f}M{j}" for f in (1, 2, 3) for j in (1, 2, 3, 4)]
    assert _map_csv_names_to_sim(_DG3F_CANON, sim) == sim


def test_dg3f_csv_matches_gripper_prefixed_asset():
    # gripper_*.usd: 'gripper_f1m1_joint' … 'gripper_f3m4_joint'.
    sim = [f"gripper_f{f}m{j}_joint" for f in (1, 2, 3) for j in (1, 2, 3, 4)]
    assert _map_csv_names_to_sim(_DG3F_CSV, sim) == sim


def test_dg3f_canon_matches_gripper_prefixed_asset():
    # gripper_*.usd: 'gripper_f1m1_joint' … 'gripper_f3m4_joint'.
    sim = [f"gripper_f{f}m{j}_joint" for f in (1, 2, 3) for j in (1, 2, 3, 4)]
    assert _map_csv_names_to_sim(_DG3F_CANON, sim) == sim


def test_mapping_is_by_name_not_position():
    # 트윈 DOF 순서가 CSV 컬럼 순서와 달라도 이름으로 대응돼야 한다.
    sim = [f"F{f}M{j}" for f in (1, 2, 3) for j in (1, 2, 3, 4)]
    shuffled = sim[::-1]
    out = _map_csv_names_to_sim(_DG3F_CANON, shuffled)
    # 반환은 csv_names 순서, 값은 대응되는 트윈 DOF 이름.
    assert out == [f"F{f}M{j}" for f in (1, 2, 3) for j in (1, 2, 3, 4)]
    assert set(out) == set(shuffled)


def test_unmatchable_names_raise_with_hint():
    try:
        _map_csv_names_to_sim(_DG3F_CANON, ["nope"] * 12)
    except ValueError as e:
        assert "DG-3F->FNMM" in str(e)
    else:
        raise AssertionError("매칭 실패 시 ValueError 가 나야 함")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
