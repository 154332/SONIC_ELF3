import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation, Slerp


INPUT = "Neutral_walk_forward_002__A057.csv"
OUTPUT = "Neutral_stand_to_walk_002__A057.csv"

SOURCE_FPS = 120

STAND_SECONDS = 1.0
TRANSITION_SECONDS = 1.0

N_STAND = int(STAND_SECONDS * SOURCE_FPS)
N_TRANS = int(TRANSITION_SECONDS * SOURCE_FPS)

joint_names = [
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


df = pd.read_csv(INPUT)

walk0 = df.iloc[0].copy()

# ------------------------------------------------
# 1. 构造 ELF3 standing pose
# ------------------------------------------------

stand = walk0.copy()

# 默认全部关节 0 degree
for name in joint_names:
    stand[name] = 0.0


RAD2DEG = 180.0 / np.pi

# -0.15 / +0.30 / -0.15 rad
for side in ["l", "r"]:
    stand[f"{side}_hip_y_joint"] = -0.15 * RAD2DEG
    stand[f"{side}_knee_y_joint"] = 0.30 * RAD2DEG
    stand[f"{side}_ankle_y_joint"] = -0.15 * RAD2DEG


# ------------------------------------------------
# 2. Root standing pose
#
# CSV root translation单位为 cm
# ------------------------------------------------

# 保持和 walk frame0 相同的世界 XY
stand["root_translateX"] = walk0["root_translateX"]
stand["root_translateY"] = walk0["root_translateY"]

# 1.08 m -> 108 cm
stand["root_translateZ"] = 108.0

# torso upright
stand["root_rotateX"] = 0.0
stand["root_rotateY"] = 0.0

# 保持和 walking 相同朝向
stand["root_rotateZ"] = walk0["root_rotateZ"]


# ------------------------------------------------
# 3. 1秒 static stand
# ------------------------------------------------

stand_frames = pd.DataFrame(
    [stand.copy() for _ in range(N_STAND)]
)


# ------------------------------------------------
# 4. stand -> walk frame0 transition
# ------------------------------------------------

r_stand = Rotation.from_euler(
    "xyz",
    [
        stand["root_rotateX"],
        stand["root_rotateY"],
        stand["root_rotateZ"],
    ],
    degrees=True,
)

r_walk = Rotation.from_euler(
    "xyz",
    [
        walk0["root_rotateX"],
        walk0["root_rotateY"],
        walk0["root_rotateZ"],
    ],
    degrees=True,
)

rotations = Rotation.from_quat(
    np.vstack([
        r_stand.as_quat(),
        r_walk.as_quat(),
    ])
)

slerp = Slerp(
    [0.0, 1.0],
    rotations,
)

transition = []

for k in range(N_TRANS):

    t = k / N_TRANS

    # fifth-order smootherstep
    s = 6*t**5 - 15*t**4 + 10*t**3

    row = stand.copy()

    # root translation
    for name in [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
    ]:
        row[name] = (
            (1.0 - s) * stand[name]
            + s * walk0[name]
        )

    # all 31 joint positions
    for name in joint_names:
        row[name] = (
            (1.0 - s) * stand[name]
            + s * walk0[name]
        )

    # root orientation via SLERP
    euler = slerp([s]).as_euler(
        "xyz",
        degrees=True,
    )[0]

    row["root_rotateX"] = euler[0]
    row["root_rotateY"] = euler[1]
    row["root_rotateZ"] = euler[2]

    transition.append(row)


transition = pd.DataFrame(transition)


# ------------------------------------------------
# 5. concatenate
# ------------------------------------------------

result = pd.concat(
    [
        stand_frames,
        transition,
        df,
    ],
    ignore_index=True,
)

# rebuild frame number
result["Frame"] = np.arange(len(result))

result.to_csv(
    OUTPUT,
    index=False,
)

print("Saved:", OUTPUT)
print("Original:", len(df))
print("Stand:", len(stand_frames))
print("Transition:", len(transition))
print("Total:", len(result))