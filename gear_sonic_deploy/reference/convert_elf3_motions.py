#!/usr/bin/env python3
"""Convert ELF3 motion pickle data to SONIC C++ deployment reference format."""

import argparse
import pickle
from pathlib import Path
import copy

import numpy as np

ELF3_NUM_DOF=31
ELF3_NUM_BODY=32
TARGET_FPS=50.0

ELF3_ISAACLAB_TO_MUJOCO_DOF=[3,7,10,13,17,21,25,27,29,14,18,22,26,28,30,1,5,8,11,15,19,23,2,6,9,12,16,20,24,0,4]
ELF3_MUJOCO_TO_ISAACLAB_DOF=[29,15,22,0,30,16,23,1,17,24,2,18,25,3,9,19,26,4,10,20,27,5,11,21,28,6,12,7,13,8,14]
ELF3_ISAACLAB_TO_MUJOCO_BODY=[0,4,8,11,14,18,22,26,28,30,15,19,23,27,29,31,2,6,9,12,16,20,24,3,7,10,13,17,21,25,1,5]
ELF3_MUJOCO_TO_ISAACLAB_BODY=[0,30,16,23,1,31,17,24,2,18,25,3,19,26,4,10,20,27,5,11,21,28,6,12,22,29,7,13,8,14,9,15]

REQUIRED_KEYS=["joint_pos","joint_vel","body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"]


def to_numpy(value):
    if isinstance(value,np.ndarray):
        return value
    if hasattr(value,"detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_pickle(path):
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        with open(path,"rb") as f:
            return pickle.load(f)


def infer_source_fps(motion_data,cli_fps):
    if cli_fps is not None:
        return float(cli_fps)
    for key in ["fps","frame_rate","framerate"]:
        if key in motion_data:
            value=motion_data[key]
            if hasattr(value,"item"):
                value=value.item()
            return float(value)
    if "dt" in motion_data:
        value=motion_data["dt"]
        if hasattr(value,"item"):
            value=value.item()
        return 1.0/float(value)
    print("  WARNING: source FPS not found, assuming 50 Hz")
    return TARGET_FPS


def make_time_axes(num_frames,source_fps,target_fps):
    if num_frames<=1:
        return np.zeros(num_frames),np.zeros(num_frames)
    duration=(num_frames-1)/source_fps
    target_frames=int(round(duration*target_fps))+1
    src_t=np.arange(num_frames,dtype=np.float64)/source_fps
    dst_t=np.arange(target_frames,dtype=np.float64)/target_fps
    dst_t=np.clip(dst_t,0.0,duration)
    return src_t,dst_t


def resample_linear(array,source_fps,target_fps):
    array=np.asarray(array)
    if array.shape[0]<=1 or abs(source_fps-target_fps)<1e-9:
        return array.copy()
    src_t,dst_t=make_time_axes(array.shape[0],source_fps,target_fps)
    original_shape=array.shape
    flat=array.reshape(array.shape[0],-1)
    result=np.empty((len(dst_t),flat.shape[1]),dtype=np.float64)
    for i in range(flat.shape[1]):
        result[:,i]=np.interp(dst_t,src_t,flat[:,i])
    return result.reshape((len(dst_t),)+original_shape[1:]).astype(array.dtype)


def resample_quat_wxyz(array,source_fps,target_fps):
    array=np.asarray(array)
    if array.shape[0]<=1 or abs(source_fps-target_fps)<1e-9:
        return array.copy()
    from scipy.spatial.transform import Rotation,Slerp
    src_t,dst_t=make_time_axes(array.shape[0],source_fps,target_fps)
    result=np.empty((len(dst_t),array.shape[1],4),dtype=np.float64)
    for body_idx in range(array.shape[1]):
        q=array[:,body_idx,:].astype(np.float64)
        q_xyzw=q[:,[1,2,3,0]]
        q_xyzw/=np.maximum(np.linalg.norm(q_xyzw,axis=1,keepdims=True),1e-12)
        rotations=Rotation.from_quat(q_xyzw)
        interpolated=Slerp(src_t,rotations)(dst_t).as_quat()
        result[:,body_idx,:]=interpolated[:,[3,0,1,2]]
    return result.astype(array.dtype)


def convert_quat_to_wxyz(array,input_order):
    if input_order=="wxyz":
        return array
    return array[..., [3,0,1,2]]


def convert_joint_order(array,input_order):
    if input_order=="isaaclab":
        return array
    return array[:,ELF3_MUJOCO_TO_ISAACLAB_DOF]


def convert_body_data_order(array,body_indexes,input_order):
    if input_order=="isaaclab":
        return array,body_indexes
    body_count=array.shape[1]
    if body_count==ELF3_NUM_BODY and body_indexes is None:
        return array[:,ELF3_MUJOCO_TO_ISAACLAB_BODY,...],np.arange(ELF3_NUM_BODY,dtype=np.int64)
    if body_indexes is None:
        if body_count==1:
            return array,np.array([0],dtype=np.int64)
        raise ValueError("MuJoCo-order body data with a subset of bodies requires '_body_indexes'")
    body_indexes=np.asarray(body_indexes,dtype=np.int64).reshape(-1)
    if len(body_indexes)!=body_count:
        raise ValueError(f"_body_indexes length {len(body_indexes)} != body count {body_count}")
    converted=np.array([ELF3_ISAACLAB_TO_MUJOCO_BODY[int(i)] for i in body_indexes],dtype=np.int64)
    return array,converted


def save_array_as_csv(array,filename,headers):
    array=np.asarray(array)
    with open(filename,"w") as f:
        f.write(",".join(headers)+"\n")
        for row in array:
            f.write(",".join(f"{float(v):.8f}" for v in row)+"\n")


def save_metadata(motion_name,body_indexes,timesteps,filename):
    body_indexes=np.asarray(body_indexes,dtype=np.int64).reshape(-1)
    with open(filename,"w") as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write("="*30+"\n\n")
        f.write("Body part indexes:\n")
        f.write("["+" ".join(str(int(v)) for v in body_indexes)+"]\n\n")
        f.write(f"Total timesteps: {timesteps}\n")


def save_info(motion_name,data,filename,source_fps,target_fps):
    with open(filename,"w") as f:
        f.write(f"Motion Information: {motion_name}\n")
        f.write("="*50+"\n\n")
        f.write(f"Source FPS: {source_fps}\n")
        f.write(f"Target FPS: {target_fps}\n\n")
        for key,value in data.items():
            if isinstance(value,np.ndarray):
                f.write(f"{key}:\n")
                f.write(f"  Shape: {value.shape}\n")
                f.write(f"  Dtype: {value.dtype}\n")
                if value.size:
                    f.write(f"  Range: [{float(np.min(value)):.6f}, {float(np.max(value)):.6f}]\n")
                f.write("\n")

def motionlib_to_deploy_format(motion_name, motion_data):
    """Convert SONIC training motion_lib entry to deployment-reference format."""

    required = ["dof", "root_trans_offset", "root_rot", "fps"]
    missing = [k for k in required if k not in motion_data]
    if missing:
        raise KeyError(
            f"{motion_name}: motion_lib entry missing keys: {missing}"
        )

    # ------------------------------------------------------------------
    # 1. Joint positions
    #
    # convert_soma_csv_to_motion_lib_elf3.py stores ELF3 DOFs in
    # MuJoCo/controller order:
    #
    # waist -> left leg -> right leg -> left arm -> right arm -> head
    # ------------------------------------------------------------------
    joint_pos = to_numpy(motion_data["dof"]).astype(np.float32)

    if joint_pos.ndim != 2 or joint_pos.shape[1] != ELF3_NUM_DOF:
        raise ValueError(
            f"{motion_name}: expected dof [T,31], got {joint_pos.shape}"
        )

    fps = float(motion_data["fps"])
    dt = 1.0 / fps

    # Joint velocities in the SAME order as joint_pos.
    if joint_pos.shape[0] > 1:
        joint_vel = np.gradient(
            joint_pos,
            dt,
            axis=0,
        ).astype(np.float32)
    else:
        joint_vel = np.zeros_like(joint_pos)

    # ------------------------------------------------------------------
    # 2. Root position
    # ------------------------------------------------------------------
    root_pos = to_numpy(
        motion_data["root_trans_offset"]
    ).astype(np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            f"{motion_name}: root_trans_offset expected [T,3], "
            f"got {root_pos.shape}"
        )

    # Root-only body representation: [T, 1, 3]
    body_pos_w = root_pos[:, None, :]

    if root_pos.shape[0] > 1:
        root_lin_vel = np.gradient(
            root_pos,
            dt,
            axis=0,
        ).astype(np.float32)
    else:
        root_lin_vel = np.zeros_like(root_pos)

    body_lin_vel_w = root_lin_vel[:, None, :]

    # ------------------------------------------------------------------
    # 3. Root quaternion
    #
    # motion_lib root_rot is XYZW (scipy convention).
    # deployment body_quat_w expects WXYZ.
    # ------------------------------------------------------------------
    root_quat_xyzw = to_numpy(
        motion_data["root_rot"]
    ).astype(np.float32)

    if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
        raise ValueError(
            f"{motion_name}: root_rot expected [T,4], "
            f"got {root_quat_xyzw.shape}"
        )

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    body_quat_w = root_quat_wxyz[:, None, :]

    # ------------------------------------------------------------------
    # 4. Root angular velocity
    # ------------------------------------------------------------------
    from scipy.spatial.transform import Rotation

    T = root_quat_xyzw.shape[0]

    root_ang_vel = np.zeros((T, 3), dtype=np.float32)

    if T > 1:
        rotations = Rotation.from_quat(root_quat_xyzw)

        delta_rot = (
            rotations[:-1].inv()
            * rotations[1:]
        )

        root_ang_vel[:-1] = (
            delta_rot.as_rotvec() / dt
        ).astype(np.float32)

        root_ang_vel[-1] = root_ang_vel[-2]

    body_ang_vel_w = root_ang_vel[:, None, :]

    result = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,

        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,

        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,

        # Root body only
        "_body_indexes": np.array([0], dtype=np.int64),

        "fps": fps,
    }

    if "smpl_joints" in motion_data:
        result["smpl_joints"] = to_numpy(
            motion_data["smpl_joints"]
        ).astype(np.float32)

    return result

def validate_motion(motion_name,motion_data):
    missing=[key for key in REQUIRED_KEYS if key not in motion_data]
    if missing:
        raise KeyError(f"{motion_name}: missing required keys: {missing}")
    joint_pos=to_numpy(motion_data["joint_pos"])
    joint_vel=to_numpy(motion_data["joint_vel"])
    if joint_pos.ndim!=2 or joint_pos.shape[1]!=ELF3_NUM_DOF:
        raise ValueError(f"{motion_name}: joint_pos expected [T,31], got {joint_pos.shape}")
    if joint_vel.shape!=joint_pos.shape:
        raise ValueError(f"{motion_name}: joint_vel shape {joint_vel.shape} != joint_pos shape {joint_pos.shape}")
    timesteps=joint_pos.shape[0]
    for key in ["body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"]:
        value=to_numpy(motion_data[key])
        if value.shape[0]!=timesteps:
            raise ValueError(f"{motion_name}: {key} frame count {value.shape[0]} != {timesteps}")
    return timesteps


def convert_single_motion(motion_name,motion_data,output_dir,args):
    print(f"\nProcessing: {motion_name}")
    validate_motion(motion_name,motion_data)
    source_fps=infer_source_fps(motion_data,args.source_fps)
    joint_pos=to_numpy(motion_data["joint_pos"]).astype(np.float32)
    joint_vel=to_numpy(motion_data["joint_vel"]).astype(np.float32)
    body_pos=to_numpy(motion_data["body_pos_w"]).astype(np.float32)
    body_quat=to_numpy(motion_data["body_quat_w"]).astype(np.float32)
    body_lin_vel=to_numpy(motion_data["body_lin_vel_w"]).astype(np.float32)
    body_ang_vel=to_numpy(motion_data["body_ang_vel_w"]).astype(np.float32)
    if body_pos.ndim!=3 or body_pos.shape[2]!=3:
        raise ValueError(f"body_pos_w expected [T,B,3], got {body_pos.shape}")
    if body_quat.ndim!=3 or body_quat.shape[2]!=4:
        raise ValueError(f"body_quat_w expected [T,B,4], got {body_quat.shape}")
    if body_lin_vel.shape!=body_pos.shape or body_ang_vel.shape!=body_pos.shape:
        raise ValueError("body velocity arrays must have the same shape as body_pos_w")
    body_indexes=motion_data.get("_body_indexes",None)
    if body_indexes is not None:
        body_indexes=to_numpy(body_indexes).astype(np.int64).reshape(-1)
    joint_pos=convert_joint_order(joint_pos,args.input_order)
    joint_vel=convert_joint_order(joint_vel,args.input_order)
    body_quat=convert_quat_to_wxyz(body_quat,args.quat_order)
    if args.input_order=="mujoco":
        original_indexes=None if body_indexes is None else body_indexes.copy()
        body_pos,body_indexes=convert_body_data_order(body_pos,original_indexes,args.input_order)
        body_quat,_=convert_body_data_order(body_quat,original_indexes,args.input_order)
        body_lin_vel,_=convert_body_data_order(body_lin_vel,original_indexes,args.input_order)
        body_ang_vel,_=convert_body_data_order(body_ang_vel,original_indexes,args.input_order)
    elif body_indexes is None:
        if body_pos.shape[1]==1:
            body_indexes=np.array([0],dtype=np.int64)
        elif body_pos.shape[1]==ELF3_NUM_BODY:
            body_indexes=np.arange(ELF3_NUM_BODY,dtype=np.int64)
        else:
            raise ValueError("IsaacLab-order body subset requires '_body_indexes'")
    if abs(source_fps-args.target_fps)>1e-9:
        print(f"  Resampling: {source_fps:g} Hz -> {args.target_fps:g} Hz")
        joint_pos=resample_linear(joint_pos,source_fps,args.target_fps)
        joint_vel=resample_linear(joint_vel,source_fps,args.target_fps)
        body_pos=resample_linear(body_pos,source_fps,args.target_fps)
        body_quat=resample_quat_wxyz(body_quat,source_fps,args.target_fps)
        body_lin_vel=resample_linear(body_lin_vel,source_fps,args.target_fps)
        body_ang_vel=resample_linear(body_ang_vel,source_fps,args.target_fps)
    timesteps=joint_pos.shape[0]
    output_dir.mkdir(parents=True,exist_ok=True)
    save_array_as_csv(joint_pos,output_dir/"joint_pos.csv",[f"joint_{i}" for i in range(ELF3_NUM_DOF)])
    save_array_as_csv(joint_vel,output_dir/"joint_vel.csv",[f"joint_vel_{i}" for i in range(ELF3_NUM_DOF)])
    body_count=body_pos.shape[1]
    save_array_as_csv(body_pos.reshape(timesteps,-1),output_dir/"body_pos.csv",[f"body_{i}_{axis}" for i in range(body_count) for axis in "xyz"])
    save_array_as_csv(body_quat.reshape(timesteps,-1),output_dir/"body_quat.csv",[f"body_{i}_{axis}" for i in range(body_count) for axis in ["w","x","y","z"]])
    save_array_as_csv(body_lin_vel.reshape(timesteps,-1),output_dir/"body_lin_vel.csv",[f"body_{i}_vel_{axis}" for i in range(body_count) for axis in "xyz"])
    save_array_as_csv(body_ang_vel.reshape(timesteps,-1),output_dir/"body_ang_vel.csv",[f"body_{i}_angvel_{axis}" for i in range(body_count) for axis in "xyz"])
    if "smpl_joints" in motion_data:
        smpl=to_numpy(motion_data["smpl_joints"])
        if smpl.shape[0]==validate_motion(motion_name,motion_data):
            smpl=resample_linear(smpl,source_fps,args.target_fps)
            save_array_as_csv(smpl.reshape(timesteps,-1),output_dir/"smpl_joint.csv",[f"smpl_joint_{i}" for i in range(smpl.reshape(timesteps,-1).shape[1])])
    if "smpl_pose" in motion_data:
        smpl_pose=to_numpy(motion_data["smpl_pose"])
        smpl_pose=resample_linear(smpl_pose,source_fps,args.target_fps)
        save_array_as_csv(smpl_pose.reshape(timesteps,-1),output_dir/"smpl_pose.csv",[f"smpl_pose_{i}" for i in range(smpl_pose.reshape(timesteps,-1).shape[1])])
    save_metadata(motion_name,body_indexes,timesteps,output_dir/"metadata.txt")
    converted={"joint_pos":joint_pos,"joint_vel":joint_vel,"body_pos_w":body_pos,"body_quat_w":body_quat,"body_lin_vel_w":body_lin_vel,"body_ang_vel_w":body_ang_vel}
    save_info(motion_name,converted,output_dir/"info.txt",source_fps,args.target_fps)
    print(f"  Frames: {timesteps}")
    print(f"  Joints: {joint_pos.shape[1]}")
    print(f"  Bodies: {body_count}")
    print(f"  Output: {output_dir}")
    return True


def create_summary(results,output_dir,args):
    with open(output_dir/"motion_summary.txt","w") as f:
        f.write("ELF3 SONIC Deployment Motion Summary\n")
        f.write("="*40+"\n\n")
        f.write(f"Target FPS: {args.target_fps}\n")
        f.write(f"Input order: {args.input_order}\n")
        f.write(f"Quaternion input order: {args.quat_order}\n")
        f.write(f"ELF3 DOF: {ELF3_NUM_DOF}\n\n")
        for name,ok in results.items():
            f.write(f"{name}: {'OK' if ok else 'FAILED'}\n")


def parse_args():
    parser=argparse.ArgumentParser(description="Convert ELF3 reference motions for SONIC C++ deployment")
    parser.add_argument("pkl_file",type=Path)
    parser.add_argument("output_dir",type=Path,nargs="?",default=None)
    parser.add_argument("--input-order",choices=["isaaclab","mujoco"],default="isaaclab",help="Joint/body ordering in the input PKL")
    parser.add_argument("--quat-order",choices=["wxyz","xyzw"],default="wxyz",help="Quaternion ordering in body_quat_w")
    parser.add_argument("--source-fps",type=float,default=None,help="Source FPS. If omitted, infer from PKL and otherwise assume 50")
    parser.add_argument("--target-fps",type=float,default=TARGET_FPS)
    return parser.parse_args()


def main():
    args=parse_args()
    if not args.pkl_file.is_file():
        raise FileNotFoundError(args.pkl_file)
    script_dir=Path(__file__).resolve().parent
    output_dir=args.output_dir if args.output_dir is not None else script_dir/"elf3"
    output_dir=output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    print("ELF3 SONIC Motion Data Converter")
    print("================================")
    print(f"Input: {args.pkl_file}")
    print(f"Output: {output_dir}")
    print(f"Input order: {args.input_order}")
    print(f"Target FPS: {args.target_fps:g}")
    data=load_pickle(args.pkl_file)
    if not isinstance(data,dict):
        raise TypeError(f"Expected top-level dict, got {type(data)}")
    print(f"Found {len(data)} motion sequences")
    results={}   
    for motion_name, motion_data in data.items():
        try:

            local_args = copy.copy(args)

            # ----------------------------------------------------------
            # Deployment-ready PKL
            # ----------------------------------------------------------
            if all(k in motion_data for k in REQUIRED_KEYS):
                deploy_data = motion_data

            # ----------------------------------------------------------
            # SONIC training motion_lib PKL
            # ----------------------------------------------------------
            elif all(
                k in motion_data
                for k in ["dof", "root_trans_offset", "root_rot", "fps"]
            ):
                print(
                    f"  Detected SONIC motion_lib format for {motion_name}"
                )

                deploy_data = motionlib_to_deploy_format(
                    motion_name,
                    motion_data,
                )

                # dof generated by convert_soma_csv_to_motion_lib_elf3.py
                # is in MuJoCo/controller order.
                local_args.input_order = "mujoco"

                # Adapter already converts root quaternion XYZW -> WXYZ.
                local_args.quat_order = "wxyz"

                local_args.source_fps = float(motion_data["fps"])

            else:
                raise ValueError(
                    f"{motion_name}: unsupported motion format. "
                    f"Keys = {list(motion_data.keys())}"
                )

            results[motion_name] = convert_single_motion(
                motion_name,
                deploy_data,
                output_dir / motion_name,
                local_args,
            )

        except Exception as e:
            results[motion_name] = False
            print(f"  ERROR: {e}")
    create_summary(results,output_dir,args)
    success=sum(results.values())
    print(f"\nConverted {success}/{len(results)} motions")
    if success!=len(results):
        raise SystemExit(1)


if __name__=="__main__":
    main()