"""Build a separate kinematic stand-to-walk reference, preserving the source CSV."""
import ast
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/elf3_motions/csv/Neutral_walk_forward_002__A057.csv'
FPS = 120
STAND_SECONDS = 2
START_SECONDS = 2


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with SOURCE.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        raw = np.asarray([[float(v) for v in row] for row in reader])
    tree = ast.parse((ROOT / 'gear_sonic/data_process/convert_soma_csv_to_motion_lib_elf3.py').read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'ELF3_JOINT_NAMES' for t in n.targets))
    assert header[:7] == ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
                          'root_rotateX', 'root_rotateY', 'root_rotateZ']
    assert header[7:] == names and np.isfinite(raw).all()
    assert np.array_equal(raw[:, 0], np.arange(len(raw)))
    source_time = np.arange(len(raw)) / FPS
    # Interpolate actual source positions; do not independently invent velocity fields.
    positional = np.column_stack((raw[:, 1:4] / 100, np.deg2rad(raw[:, 7:])))
    curve = CubicHermiteSpline(source_time, positional,
                              np.gradient(positional, 1/FPS, axis=0, edge_order=2))
    orientations = Slerp(source_time, Rotation.from_euler('xyz', raw[:, 4:7], degrees=True))
    stand = np.zeros(positional.shape[1])
    stand[:3] = positional[0, :3]
    for side in ['l', 'r']:
        for joint, value in [('hip_y', -.15), ('knee_y', .30), ('ankle_y', -.15)]:
            stand[3 + names.index(f'{side}_{joint}_joint')] = value
    stand_rotation = Rotation.from_euler('xyz', [0, 0, raw[0, 6]], degrees=True)
    u = np.linspace(0, 1, START_SECONDS * FPS + 1)
    alpha = 6*u**5 - 15*u**4 + 10*u**3
    # Source playback speed rises smoothly from 0 to 1: v=3u²−2u³.
    # Integrating gives source phase tau=T(u³−u⁴/2), ending at source t=1s.
    tau = START_SECONDS * (u**3 - .5*u**4)
    assert tau[-1] < source_time[-1]
    moving = curve(tau)
    transition = (1-alpha[:, None])*stand + alpha[:, None]*moving
    walk_rot = orientations(tau)
    transition_rot = stand_rotation * Rotation.from_rotvec(
        (stand_rotation.inv()*walk_rot).as_rotvec()*alpha[:, None])
    consumed = int(round(tau[-1]*FPS))
    after = positional[consumed+1:]
    n_stand = STAND_SECONDS * FPS
    states = np.vstack((np.tile(stand, (n_stand, 1)), transition, after))
    quats = np.vstack((np.tile(stand_rotation.as_quat(), (n_stand, 1)),
                       transition_rot.as_quat(),
                       Rotation.from_euler('xyz', raw[consumed+1:, 4:7], degrees=True).as_quat()))

    # Kinematic ground alignment: ensure neither foot capsule penetrates z=0,
    # and the lowest capsule is on z=0. This is not contact dynamics or foot IK.
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'gear_sonic_deploy/elf3/mjcf/elf3.xml'))
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    assert all(j >= 0 for j in joint_ids)
    qadr = model.jnt_qposadr[joint_ids]
    feet = {side: [g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith(side+'_foot')]
            for side in ['l', 'r']}
    assert all(feet.values())
    assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for gs in feet.values() for g in gs)
    heights, correction = [], []
    for k, state in enumerate(states):
        data.qpos[:3] = state[:3]
        data.qpos[3:7] = quats[k, [3, 0, 1, 2]]
        data.qpos[qadr] = state[3:]
        mujoco.mj_forward(model, data)
        z = [min(data.geom_xpos[g, 2] - model.geom_size[g, 0]
                 - model.geom_size[g, 1]*abs(data.geom_xmat[g].reshape(3, 3)[2, 2])
                 for g in feet[side]) for side in ['l', 'r']]
        shift = -min(z)
        states[k, 2] += shift
        correction.append(shift)
        heights.append(np.asarray(z)+shift)
    heights = np.asarray(heights)
    values = np.column_stack((np.arange(len(states)), states[:, :3]*100,
                             Rotation.from_quat(quats).as_euler('xyz', degrees=True),
                             np.rad2deg(states[:, 3:])))
    assert np.isfinite(values).all()
    assert np.allclose(np.diff(values[:n_stand+1, 1:], axis=0), 0, atol=1e-10)
    assert heights.min() >= -1e-9
    limits = model.jnt_range[joint_ids]
    bad = (states[:, 3:] < limits[:, 0]-1e-7) | (states[:, 3:] > limits[:, 1]+1e-7)
    violations = {name: int(bad[:, j].sum()) for j, name in enumerate(names) if bad[:, j].any()}
    assert not violations, f'Joint limits exceeded: {violations}'
    csv_dir = OUT / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    destination = csv_dir / 'Neutral_stand_to_walk_002__A057.csv'
    with destination.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in values:
            writer.writerow([int(row[0])] + [f'{x:.12f}' for x in row[1:]])
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    # Exercise the existing converter without editing it or producing a training PKL.
    import importlib.util
    spec = importlib.util.spec_from_file_location('elf3_converter', ROOT / 'gear_sonic/data_process/convert_soma_csv_to_motion_lib_elf3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.convert_motion(str(destination), FPS, 30)
    assert np.max(abs(np.diff(converted['dof'][:61], axis=0))) == 0
    report = {
        'source': str(SOURCE), 'source_sha256': source_hash,
        'csv': str(destination), 'fps': FPS, 'frames': len(states),
        'duration_seconds': (len(states)-1)/FPS, 'columns': len(header),
        'stand_seconds': STAND_SECONDS, 'startup_seconds': START_SECONDS,
        'walk_source_time_at_startup_end_seconds': float(tau[-1]),
        'joint_limit_violations': violations, 'first_2s_pose_and_velocity_static': True,
        'minimum_foot_collision_height_m': float(heights.min()),
        'stand_foot_heights_m': heights[0].tolist(),
        'maximum_root_height_correction_m': float(np.max(np.abs(correction))),
        'converted_30fps_frames': len(converted['dof']),
        'original_source_unchanged': True,
        'physical_contact_or_dynamic_stability_verified': False,
        'limitations': 'Kinematic candidate; no foot planting/IK, contact-force or balance optimization. Root height aligned for all frames, including the walk continuation. Ground correction can introduce velocity changes at contact switches.'
    }
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
