#!/usr/bin/env python3
"""Generate an independent 2-second static ELF3 CSV; never edit existing code/data."""
import ast
import csv
import json
import math
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
CONVERTER = ROOT / 'gear_sonic/data_process/convert_soma_csv_to_motion_lib_elf3.py'
MODEL = ROOT / 'gear_sonic/data/assets/robot_description/mjcf/elf3/elf3.xml'
SOURCE_FPS = 120
SECONDS = 2


def joint_names():
    tree = ast.parse(CONVERTER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == 'ELF3_JOINT_NAMES' for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError('ELF3_JOINT_NAMES not found in converter')


def foot_geometry(model, data):
    results = {}
    for side in ['l', 'r']:
        ids = [i for i in range(model.ngeom) if
               (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith(side + '_foot')]
        if not ids:
            raise RuntimeError('Missing foot collision geometry')
        lowest = []
        bounds = []
        for i in ids:
            if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise RuntimeError('Expected capsule foot geometry')
            radius, half_length = model.geom_size[i, :2]
            axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
            extent = radius + half_length * np.abs(axis)
            lowest.append(data.geom_xpos[i, 2] - extent[2])
            bounds.append((data.geom_xpos[i] - extent, data.geom_xpos[i] + extent))
        results[side] = {'minimum_z_m': float(min(lowest)), 'bounds': bounds}
    return results


def main():
    names = joint_names()
    assert len(names) == 31 and len(set(names)) == 31
    pose = dict.fromkeys(names, 0.0)
    for side in ['l', 'r']:
        pose[f'{side}_hip_y_joint'] = -0.15
        pose[f'{side}_knee_y_joint'] = 0.30
        pose[f'{side}_ankle_y_joint'] = -0.15

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 1.08, 1, 0, 0, 0]
    for name, angle in pose.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise RuntimeError(f'Missing joint: {name}')
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            assert lo <= angle <= hi, (name, angle, lo, hi)
        data.qpos[model.jnt_qposadr[j]] = angle
    mujoco.mj_forward(model, data)
    initial = foot_geometry(model, data)
    assert abs(initial['l']['minimum_z_m'] - initial['r']['minimum_z_m']) < 1e-8
    data.qpos[2] -= min(v['minimum_z_m'] for v in initial.values())
    mujoco.mj_forward(model, data)
    grounded = foot_geometry(model, data)
    assert all(abs(v['minimum_z_m']) < 1e-8 for v in grounded.values())
    com = data.subtree_com[1].copy()
    bounds = [b for v in grounded.values() for b in v['bounds']]
    minimum = np.min([b[0] for b in bounds], axis=0)
    maximum = np.max([b[1] for b in bounds], axis=0)
    assert np.all(com[:2] >= minimum[:2]) and np.all(com[:2] <= maximum[:2])

    columns = ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
               'root_rotateX', 'root_rotateY', 'root_rotateZ'] + names
    # Positions in cm, Euler angles and joint angles in degrees, matching converter.
    state = [0.0, 0.0, float(data.qpos[2] * 100), 0.0, 0.0, 0.0]
    state += [math.degrees(pose[name]) for name in names]
    output = OUTPUT / 'Neutral_stand_2s_elf3.csv'
    frames = SOURCE_FPS * SECONDS + 1  # Includes both t=0 and t=2 endpoints.
    with output.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for frame in range(frames):
            writer.writerow([frame] + [format(value, '.12f') for value in state])
    with output.open() as file:
        rows = list(csv.reader(file))
    assert len(rows) == frames + 1 and all(len(row) == 38 for row in rows)
    values = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    assert np.isfinite(values).all() and np.all(np.diff(values, axis=0) == 0)
    report = {
        'csv': output.name, 'source_fps': SOURCE_FPS, 'frames': frames,
        'timestamp_span_seconds': (frames - 1) / SOURCE_FPS,
        'columns': len(columns), 'joints': len(names),
        'root_height_m': float(data.qpos[2]),
        'foot_minimum_z_m': {s: grounded[s]['minimum_z_m'] for s in ['l', 'r']},
        'com_m': com.tolist(),
        'all_joint_limits_passed': True,
        'all_pose_frames_identical': True,
        'finite_difference_reference_velocities_zero': True,
        'dynamic_stability_tested': False,
        'requested_walk_source_csv_available': False,
    }
    (OUTPUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
