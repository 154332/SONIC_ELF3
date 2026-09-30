#!/usr/bin/env python3
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import transform
NUM_DOF=31
NUM_BODIES=32
ELF3_JOINT_NAMES=["waist_y_joint","waist_x_joint","waist_z_joint","l_hip_y_joint","l_hip_x_joint","l_hip_z_joint","l_knee_y_joint","l_ankle_y_joint","l_ankle_x_joint","r_hip_y_joint","r_hip_x_joint","r_hip_z_joint","r_knee_y_joint","r_ankle_y_joint","r_ankle_x_joint","l_shoulder_y_joint","l_shoulder_x_joint","l_shoulder_z_joint","l_elbow_y_joint","l_wrist_x_joint","l_wrist_y_joint","l_wrist_z_joint","r_shoulder_y_joint","r_shoulder_x_joint","r_shoulder_z_joint","r_elbow_y_joint","r_wrist_x_joint","r_wrist_y_joint","r_wrist_z_joint","head_z_joint","head_y_joint"]
DOF_AXIS=np.array([[0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,0,1],[0,1,0],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,1,0],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1],[0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0]],dtype=np.float32)
def load_elf3_csv(csv_path):
    data=pd.read_csv(csv_path)
    missing=[name for name in ELF3_JOINT_NAMES if name not in data.columns]
    if missing: raise ValueError(f"Missing ELF3 joints: {missing}")
    root_pos=data[["root_translateX","root_translateY","root_translateZ"]].to_numpy(dtype=np.float32)/100.0
    euler_deg=data[["root_rotateX","root_rotateY","root_rotateZ"]].to_numpy(dtype=np.float64)
    root_quat_xyzw=transform.Rotation.from_euler("xyz",euler_deg,degrees=True).as_quat().astype(np.float32)
    dof=np.deg2rad(data[ELF3_JOINT_NAMES].to_numpy(dtype=np.float32)).astype(np.float32)
    return root_pos,root_quat_xyzw,dof
def convert_motion(csv_path,fps_source,fps_target):
    root_pos,root_quat_xyzw,dof=load_elf3_csv(csv_path)
    T=dof.shape[0]
    if dof.shape[1]!=NUM_DOF: raise ValueError(f"Expected {NUM_DOF} DOFs, got {dof.shape[1]}")
    pose_aa=np.zeros((T,NUM_BODIES,3),dtype=np.float32)
    pose_aa[:,0,:]=transform.Rotation.from_quat(root_quat_xyzw).as_rotvec().astype(np.float32)
    pose_aa[:,1:,:]=DOF_AXIS[None,:,:]*dof[:,:,None]
    smpl_joints=np.zeros((T,24,3),dtype=np.float32)
    if fps_source!=fps_target:
        if fps_source%fps_target!=0: raise ValueError(f"fps_source={fps_source} must be divisible by fps={fps_target}")
        stride=fps_source//fps_target
        root_pos=root_pos[::stride]
        root_quat_xyzw=root_quat_xyzw[::stride]
        dof=dof[::stride]
        pose_aa=pose_aa[::stride]
        smpl_joints=smpl_joints[::stride]
    return {"root_trans_offset":root_pos.astype(np.float32),"pose_aa":pose_aa.astype(np.float32),"dof":dof.astype(np.float32),"root_rot":root_quat_xyzw.astype(np.float32),"smpl_joints":smpl_joints,"fps":fps_target}
def collect_csvs(path):
    files=[]
    for root,_,names in os.walk(path):
        for name in sorted(names):
            if name.lower().endswith(".csv"): files.append(os.path.join(root,name))
    return files
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--fps_source",type=int,default=120)
    parser.add_argument("--individual",action="store_true")
    parser.add_argument("--num_workers",type=int,default=1)
    args=parser.parse_args()
    os.makedirs(args.output,exist_ok=True)
    csv_files=collect_csvs(args.input)
    if not csv_files: raise RuntimeError(f"No CSV files found in {args.input}")
    print(f"ELF3 {NUM_DOF} DOFs, {NUM_BODIES} bodies")
    print(f"Found {len(csv_files)} CSV files")
    converted=0
    failed=0
    combined={}
    for csv_path in csv_files:
        name=os.path.splitext(os.path.basename(csv_path))[0]
        try:
            entry=convert_motion(csv_path,args.fps_source,args.fps)
            if args.individual:
                rel_dir=os.path.relpath(os.path.dirname(csv_path),args.input)
                out_dir=args.output if rel_dir=="." else os.path.join(args.output,rel_dir)
                os.makedirs(out_dir,exist_ok=True)
                out_path=os.path.join(out_dir,name+".pkl")
                joblib.dump({name:entry},out_path,compress=True)
                print(f"[OK] {name}: dof={entry['dof'].shape}, pose_aa={entry['pose_aa'].shape} -> {out_path}")
            else:
                combined[name]=entry
            converted+=1
        except Exception as e:
            failed+=1
            print(f"[FAILED] {csv_path}: {type(e).__name__}: {e}")
    if not args.individual:
        out_path=os.path.join(args.output,"elf3_motion_lib.pkl")
        joblib.dump(combined,out_path,compress=True)
        print(f"Saved combined PKL: {out_path}")
    print(f"Done: {converted} converted, {failed} failed")
if __name__=="__main__":
    main()
