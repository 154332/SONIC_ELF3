#!/usr/bin/env python3
"""
Offline MuJoCo visualizer for ELF3 SONIC motion_lib PKL.

Designed for PKLs produced by convert_soma_csv_to_motion_lib_elf3.py:
    {
        motion_name: {
            "root_trans_offset": (T, 3),   # meters
            "root_rot":          (T, 4),   # xyzw
            "dof":               (T, 31),  # ELF3 MuJoCo/controller joint order
            "fps":               int,
            ...
        }
    }

Controls:
    Space : pause / resume
    R     : reset to frame 0
    .     : next frame
    ,     : previous frame
"""

import argparse
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np


ELF3_JOINT_NAMES = [
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
    "r_hip_y_joint",
    "r_hip_x_joint",
    "r_hip_z_joint",
    "r_knee_y_joint",
    "r_ankle_y_joint",
    "r_ankle_x_joint",
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
    "head_z_joint",
    "head_y_joint",
]

paused = False
frame_idx = 0


def key_callback(keycode):
    global paused, frame_idx

    try:
        c = chr(keycode)
    except Exception:
        return

    if c == " ":
        paused = not paused
        print("Paused" if paused else "Playing")
    elif c in ("R", "r"):
        frame_idx = 0
        print("Reset -> frame 0")
    elif c == ".":
        paused = True
        frame_idx += 1
        print("frame", frame_idx)
    elif c == ",":
        paused = True
        frame_idx = max(0, frame_idx - 1)
        print("frame", frame_idx)


def find_free_joint_qpos_addr(model):
    """Prefer ELF3 world_joint; otherwise use the first MuJoCo free joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    if jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        return int(model.jnt_qposadr[jid]), "world_joint"

    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            return int(model.jnt_qposadr[jid]), name

    raise RuntimeError("No free/root joint found in ELF3 MuJoCo model.")


def build_joint_qpos_addresses(model):
    addrs = []
    missing = []

    for name in ELF3_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name)
            continue
        addrs.append(int(model.jnt_qposadr[jid]))

    if missing:
        raise RuntimeError(
            "ELF3 XML is missing expected joints:\n  " + "\n  ".join(missing)
        )

    if len(addrs) != 31:
        raise RuntimeError(f"Expected 31 ELF3 joint qpos addresses, got {len(addrs)}")

    return addrs


def load_motion(pkl_path, motion_key=None):
    obj = joblib.load(pkl_path)

    # Standard SONIC motion_lib layout: {motion_name: entry}
    if isinstance(obj, dict) and "dof" not in obj:
        if not obj:
            raise RuntimeError("PKL contains no motions.")

        if motion_key is None:
            motion_key = next(iter(obj.keys()))

        if motion_key not in obj:
            raise KeyError(
                f"Motion key '{motion_key}' not found. "
                f"Available keys: {list(obj.keys())}"
            )

        motion = obj[motion_key]
    else:
        motion_key = motion_key or Path(pkl_path).stem
        motion = obj

    required = ["root_trans_offset", "root_rot", "dof"]
    missing = [k for k in required if k not in motion]
    if missing:
        raise KeyError(f"Motion is missing required fields: {missing}")

    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_trans_offset must be (T,3), got {root_pos.shape}")
    if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
        raise ValueError(f"root_rot must be (T,4), got {root_quat_xyzw.shape}")
    if dof.ndim != 2 or dof.shape[1] != 31:
        raise ValueError(f"dof must be (T,31), got {dof.shape}")

    T = dof.shape[0]
    if root_pos.shape[0] != T or root_quat_xyzw.shape[0] != T:
        raise ValueError(
            "Motion lengths differ: "
            f"root_pos={root_pos.shape[0]}, "
            f"root_rot={root_quat_xyzw.shape[0]}, "
            f"dof={T}"
        )

    fps = float(motion.get("fps", 30.0))
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    return motion_key, root_pos, root_quat_xyzw, dof, fps


def set_frame(
    model,
    data,
    root_qpos_addr,
    joint_qpos_addrs,
    root_pos,
    root_quat_xyzw,
    dof,
    idx,
    root_z_offset=0.0,
):
    # Root position.
    pos = root_pos[idx].copy()
    pos[2] += root_z_offset
    data.qpos[root_qpos_addr:root_qpos_addr + 3] = pos

    # PKL converter uses scipy Rotation.as_quat(): [x, y, z, w].
    # MuJoCo free-joint qpos requires [w, x, y, z].
    q_xyzw = root_quat_xyzw[idx]
    norm = np.linalg.norm(q_xyzw)
    if norm < 1e-12:
        q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q_xyzw = q_xyzw / norm

    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    data.qpos[root_qpos_addr + 3:root_qpos_addr + 7] = q_wxyz

    # Set each ELF3 joint by NAME rather than assuming qpos layout.
    for value, qaddr in zip(dof[idx], joint_qpos_addrs):
        data.qpos[qaddr] = value

    # Kinematic visualization only: do not integrate physics.
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an ELF3 SONIC motion_lib PKL in MuJoCo."
    )
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to Neutral_stand_to_walk_002__A057.pkl or another ELF3 motion_lib PKL.",
    )
    parser.add_argument(
        "--xml",
        default="elf3/mjcf/elf3.xml",
        help="ELF3 MuJoCo XML path. Default assumes execution from gear_sonic_deploy/.",
    )
    parser.add_argument(
        "--motion-key",
        default=None,
        help="Motion key inside the PKL. If omitted, the first motion is used.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier, e.g. 0.5 or 1.0.",
    )
    parser.add_argument(
        "--root-z-offset",
        type=float,
        default=0.0,
        help="Optional visualization-only root Z offset in meters.",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    xml_path = Path(args.xml).expanduser().resolve()
    pkl_path = Path(args.pkl).expanduser().resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"ELF3 XML not found: {xml_path}")
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Motion PKL not found: {pkl_path}")

    print("Loading ELF3 model:")
    print(" ", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    root_addr, root_joint_name = find_free_joint_qpos_addr(model)
    joint_addrs = build_joint_qpos_addresses(model)

    motion_name, root_pos, root_quat_xyzw, dof, fps = load_motion(
        str(pkl_path), args.motion_key
    )

    print("\nELF3 model loaded")
    print(f"  nq={model.nq}, nv={model.nv}, njnt={model.njnt}")
    print(f"  root joint: {root_joint_name}, qpos address={root_addr}")
    print(f"  ELF3 controlled joints: {len(joint_addrs)}")

    print("\nMotion loaded")
    print(f"  name: {motion_name}")
    print(f"  frames: {dof.shape[0]}")
    print(f"  dof: {dof.shape}")
    print(f"  fps: {fps}")
    print(f"  duration: {dof.shape[0] / fps:.3f} s")
    print(f"  frame0 root: {root_pos[0]}")
    print(f"  frame0 dof: {np.array2string(dof[0], precision=4, suppress_small=True)}")

    print("\nControls:")
    print("  Space : pause/resume")
    print("  R     : reset")
    print("  .     : next frame")
    print("  ,     : previous frame")

    global frame_idx, paused
    frame_idx = 0
    paused = False

    frame_period = 1.0 / (fps * args.speed)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0

        last_advance = time.perf_counter()

        while viewer.is_running():
            T = dof.shape[0]
            frame_idx %= T

            set_frame(
                model,
                data,
                root_addr,
                joint_addrs,
                root_pos,
                root_quat_xyzw,
                dof,
                frame_idx,
                args.root_z_offset,
            )

            viewer.sync()

            now = time.perf_counter()

            if not paused and now - last_advance >= frame_period:
                # Advance one source frame at the motion's own FPS.
                frame_idx = (frame_idx + 1) % T
                last_advance = now
            else:
                # Keep the GUI responsive without busy-spinning.
                time.sleep(min(0.002, frame_period / 4.0))


if __name__ == "__main__":
    main()
