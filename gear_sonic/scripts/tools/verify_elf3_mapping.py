#!/usr/bin/env python3
import ast
import sys
import xml.etree.ElementTree as ET
import numpy as np

ELF3_PY="gear_sonic/envs/manager_env/robots/elf3.py"
XML_PATH="gear_sonic/data/assets/robot_description/mjcf/elf3/elf3.xml"

ISAACLAB_DOF_NAMES=["head_z_joint","l_shoulder_y_joint","r_shoulder_y_joint","waist_y_joint","head_y_joint","l_shoulder_x_joint","r_shoulder_x_joint","waist_x_joint","l_shoulder_z_joint","r_shoulder_z_joint","waist_z_joint","l_elbow_y_joint","r_elbow_y_joint","l_hip_y_joint","r_hip_y_joint","l_wrist_x_joint","r_wrist_x_joint","l_hip_x_joint","r_hip_x_joint","l_wrist_y_joint","r_wrist_y_joint","l_hip_z_joint","r_hip_z_joint","l_wrist_z_joint","r_wrist_z_joint","l_knee_y_joint","r_knee_y_joint","l_ankle_y_joint","r_ankle_y_joint","l_ankle_x_joint","r_ankle_x_joint"]

ISAACLAB_BODY_NAMES=["torso_link","head_z_link","l_shoulder_y_link","r_shoulder_y_link","waist_y_link","head_y_link","l_shoulder_x_link","r_shoulder_x_link","waist_x_link","l_shoulder_z_link","r_shoulder_z_link","waist_z_link","l_elbow_y_link","r_elbow_y_link","l_hip_y_link","r_hip_y_link","l_wrist_x_link","r_wrist_x_link","l_hip_x_link","r_hip_x_link","l_wrist_y_link","r_wrist_y_link","l_hip_z_link","r_hip_z_link","l_wrist_z_link","r_wrist_z_link","l_knee_y_link","r_knee_y_link","l_ankle_y_link","r_ankle_y_link","l_ankle_x_link","r_ankle_x_link"]

TARGET_NAMES=["ELF3_ISAACLAB_JOINTS","ELF3_ISAACLAB_TO_MUJOCO_DOF","ELF3_MUJOCO_TO_ISAACLAB_DOF","ELF3_ISAACLAB_TO_MUJOCO_BODY","ELF3_MUJOCO_TO_ISAACLAB_BODY"]

def load_static_values(path):
    tree=ast.parse(open(path,"r",encoding="utf-8").read(),filename=path)
    out={}
    for node in tree.body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name): continue
        name=node.targets[0].id
        if name in TARGET_NAMES: out[name]=ast.literal_eval(node.value)
    missing=[n for n in TARGET_NAMES if n not in out]
    if missing: raise RuntimeError(f"elf3.py缺少变量: {missing}")
    return out

def parse_mujoco_orders(path):
    root=ET.parse(path).getroot()
    joints=[]
    for j in root.findall(".//joint"):
        name=j.get("name")
        if not name or name=="world_joint" or j.get("type")=="free": continue
        joints.append(name)
    bodies=[b.get("name") for b in root.findall(".//worldbody//body") if b.get("name")]
    return joints,bodies

def expected_mapping(source_names,target_names):
    missing=[n for n in target_names if n not in source_names]
    if missing: raise RuntimeError(f"目标名称在源列表中不存在: {missing}")
    return [source_names.index(n) for n in target_names]

def check_mapping(title,source_names,target_names,mapping):
    print(f"\n========== {title} ==========")
    ok=True
    if len(mapping)!=len(target_names):
        print(f"[FAIL] 长度错误: mapping={len(mapping)}, expected={len(target_names)}")
        return False
    if sorted(mapping)!=list(range(len(source_names))):
        print("[FAIL] mapping不是0..N-1完整排列")
        ok=False
    for target_idx,source_idx in enumerate(mapping):
        expected=target_names[target_idx]
        actual=source_names[source_idx]
        if actual!=expected:
            print(f"[FAIL] target[{target_idx:2d}] {expected:<24} <- source[{source_idx:2d}] {actual}")
            ok=False
    if ok: print("[PASS] 所有名称逐项匹配")
    return ok

def check_roundtrip(title,n,a_to_b,b_to_a):
    x=np.arange(n)
    y=x[a_to_b]
    z=y[b_to_a]
    ok=np.array_equal(x,z)
    print(f"{title}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("original:",x.tolist())
        print("result  :",z.tolist())
    return ok

def main():
    cfg=load_static_values(ELF3_PY)
    mj_dofs,mj_bodies=parse_mujoco_orders(XML_PATH)
    il_body_cfg=cfg["ELF3_ISAACLAB_JOINTS"]
    il_to_mj_dof=cfg["ELF3_ISAACLAB_TO_MUJOCO_DOF"]
    mj_to_il_dof=cfg["ELF3_MUJOCO_TO_ISAACLAB_DOF"]
    il_to_mj_body=cfg["ELF3_ISAACLAB_TO_MUJOCO_BODY"]
    mj_to_il_body=cfg["ELF3_MUJOCO_TO_ISAACLAB_BODY"]

    print("IsaacLab DOF :",len(ISAACLAB_DOF_NAMES))
    print("MuJoCo DOF   :",len(mj_dofs))
    print("IsaacLab BODY:",len(ISAACLAB_BODY_NAMES))
    print("MuJoCo BODY  :",len(mj_bodies))

    if il_body_cfg==ISAACLAB_BODY_NAMES: print("[PASS] ELF3_ISAACLAB_JOINTS与IsaacLab BODY顺序一致")
    else:
        print("[FAIL] ELF3_ISAACLAB_JOINTS与IsaacLab BODY顺序不一致")
        for i,(a,b) in enumerate(zip(il_body_cfg,ISAACLAB_BODY_NAMES)):
            if a!=b: print(f"  [{i}] config={a}, actual={b}")

    exp_il_to_mj_dof=expected_mapping(ISAACLAB_DOF_NAMES,mj_dofs)
    exp_mj_to_il_dof=expected_mapping(mj_dofs,ISAACLAB_DOF_NAMES)
    exp_il_to_mj_body=expected_mapping(ISAACLAB_BODY_NAMES,mj_bodies)
    exp_mj_to_il_body=expected_mapping(mj_bodies,ISAACLAB_BODY_NAMES)

    results=[]
    results.append(check_mapping("ISAACLAB -> MUJOCO DOF",ISAACLAB_DOF_NAMES,mj_dofs,il_to_mj_dof))
    results.append(check_mapping("MUJOCO -> ISAACLAB DOF",mj_dofs,ISAACLAB_DOF_NAMES,mj_to_il_dof))
    results.append(check_mapping("ISAACLAB -> MUJOCO BODY",ISAACLAB_BODY_NAMES,mj_bodies,il_to_mj_body))
    results.append(check_mapping("MUJOCO -> ISAACLAB BODY",mj_bodies,ISAACLAB_BODY_NAMES,mj_to_il_body))

    print("\n========== ROUND TRIP ==========")
    results.append(check_roundtrip("DOF  Isaac -> MuJoCo -> Isaac",31,il_to_mj_dof,mj_to_il_dof))
    results.append(check_roundtrip("BODY Isaac -> MuJoCo -> Isaac",32,il_to_mj_body,mj_to_il_body))

    print("\n========== EXPECTED ==========")
    print("ELF3_ISAACLAB_TO_MUJOCO_DOF =",exp_il_to_mj_dof)
    print("ELF3_MUJOCO_TO_ISAACLAB_DOF =",exp_mj_to_il_dof)
    print("ELF3_ISAACLAB_TO_MUJOCO_BODY =",exp_il_to_mj_body)
    print("ELF3_MUJOCO_TO_ISAACLAB_BODY =",exp_mj_to_il_body)

    print("\n========== CURRENT ==========")
    print("ELF3_ISAACLAB_TO_MUJOCO_DOF =",il_to_mj_dof)
    print("ELF3_MUJOCO_TO_ISAACLAB_DOF =",mj_to_il_dof)
    print("ELF3_ISAACLAB_TO_MUJOCO_BODY =",il_to_mj_body)
    print("ELF3_MUJOCO_TO_ISAACLAB_BODY =",mj_to_il_body)

    print("\n========== RESULT ==========")
    if all(results):
        print("ALL PASS: mapping方向和顺序都正确")
        return 0
    print("FAILED: mapping存在错误，比较EXPECTED与CURRENT")
    return 1

if __name__=="__main__":
    sys.exit(main())