from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils


ASSET_DIR = "/home/robot/GR00T-WholeBodyControl/gear_sonic/data/assets"

# ELF3 joint-side equivalent inertias [kg*m^2].
# Use the joint-level values published for ELF3 rather than the standalone
# actuator inertia when a parallel/coupled mechanism changes the equivalent
# joint-space inertia.
ARMATURE_BXI8515 = 0.044688
ARMATURE_BXI7010 = 0.0137351
ARMATURE_BXI5014 = 0.00424198

ARMATURE_WAIST_Y = 0.0274702
ARMATURE_WAIST_X = 0.0412054
ARMATURE_ANKLE_Y = 0.00848397
ARMATURE_ANKLE_X = 0.00551458

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


STIFFNESS_BXI8515 = _stiffness(ARMATURE_BXI8515)
STIFFNESS_BXI7010 = _stiffness(ARMATURE_BXI7010)
STIFFNESS_BXI5014 = _stiffness(ARMATURE_BXI5014)
STIFFNESS_WAIST_Y = _stiffness(ARMATURE_WAIST_Y)
STIFFNESS_WAIST_X = _stiffness(ARMATURE_WAIST_X)
STIFFNESS_ANKLE_Y = _stiffness(ARMATURE_ANKLE_Y)
STIFFNESS_ANKLE_X = _stiffness(ARMATURE_ANKLE_X)

DAMPING_BXI8515 = _damping(ARMATURE_BXI8515)
DAMPING_BXI7010 = _damping(ARMATURE_BXI7010)
DAMPING_BXI5014 = _damping(ARMATURE_BXI5014)
DAMPING_WAIST_Y = _damping(ARMATURE_WAIST_Y)
DAMPING_WAIST_X = _damping(ARMATURE_WAIST_X)
DAMPING_ANKLE_Y = _damping(ARMATURE_ANKLE_Y)
DAMPING_ANKLE_X = _damping(ARMATURE_ANKLE_X)


ELF3_ISAACLAB_JOINTS = [
    "torso_link",
    "head_z_link",
    "l_shoulder_y_link",
    "r_shoulder_y_link",
    "waist_y_link",
    "head_y_link",
    "l_shoulder_x_link",
    "r_shoulder_x_link",
    "waist_x_link",
    "l_shoulder_z_link",
    "r_shoulder_z_link",
    "waist_z_link",
    "l_elbow_y_link",
    "r_elbow_y_link",
    "l_hip_y_link",
    "r_hip_y_link",
    "l_wrist_x_link",
    "r_wrist_x_link",
    "l_hip_x_link",
    "r_hip_x_link",
    "l_wrist_y_link",
    "r_wrist_y_link",
    "l_hip_z_link",
    "r_hip_z_link",
    "l_wrist_z_link",
    "r_wrist_z_link",
    "l_knee_y_link",
    "r_knee_y_link",
    "l_ankle_y_link",
    "r_ankle_y_link",
    "l_ankle_x_link",
    "r_ankle_x_link",
]

# ELF3_ISAACLAB_TO_MUJOCO_DOF = [
#     29,15,22,0,30,16,23,1,17,24,
#     2,18,25,3,9,19,26,4,10,20,
#     27,5,11,21,28,6,12,7,13,8,14,
# ]

# ELF3_MUJOCO_TO_ISAACLAB_DOF = [
#     3,7,10,13,17,21,25,27,29,14,
#     18,22,26,28,30,1,5,8,11,15,
#     19,23,2,6,9,12,16,20,24,0,4,
# ]
ELF3_ISAACLAB_TO_MUJOCO_DOF = [
    3,7,10,13,17,21,25,27,29,14,
    18,22,26,28,30,1,5,8,11,15,
    19,23,2,6,9,12,16,20,24,0,4,
]
ELF3_MUJOCO_TO_ISAACLAB_DOF = [
    29,15,22,0,30,16,23,1,17,24,
    2,18,25,3,9,19,26,4,10,20,
    27,5,11,21,28,6,12,7,13,8,14,
]



# ELF3_ISAACLAB_TO_MUJOCO_BODY = [
#     0,30,16,23,1,31,17,24,
#     2,18,25,3,19,26,4,10,
#     20,27,5,11,21,28,6,12,
#     22,29,7,13,8,14,9,15,
# ]

# ELF3_MUJOCO_TO_ISAACLAB_BODY = [
#     0,4,8,11,14,18,22,26,
#     28,30,15,19,23,27,29,31,
#     2,6,9,12,16,20,24,3,
#     7,10,13,17,21,25,1,5,
# ]
ELF3_ISAACLAB_TO_MUJOCO_BODY = [
    0,4,8,11,14,18,22,26,
    28,30,15,19,23,27,29,31,
    2,6,9,12,16,20,24,3,
    7,10,13,17,21,25,1,5,
]
ELF3_MUJOCO_TO_ISAACLAB_BODY = [
    0,30,16,23,1,31,17,24,
    2,18,25,3,19,26,4,10,
    20,27,5,11,21,28,6,12,
    22,29,7,13,8,14,9,15,
]



ELF3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": ELF3_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": ELF3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": ELF3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": ELF3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": ELF3_MUJOCO_TO_ISAACLAB_BODY,
}

ELF3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/elf3/elf3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.08),

        joint_pos={
            # Left leg
            "l_hip_y_joint": -0.15,
            "l_knee_y_joint": 0.30,
            "l_ankle_y_joint": -0.15,

            # Right leg
            "r_hip_y_joint": -0.15,
            "r_knee_y_joint": 0.30,
            "r_ankle_y_joint": -0.15,

            # Right & Left leg
            # "l_elbow_y_link": 1.5,            
            # "r_elbow_y_link": 1.5,           

            # Waist
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_z_joint": 0.0,

            # Head
            "head_z_joint": 0.0,
            "head_y_joint": 0.0,
        },

        joint_vel={
            ".*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Legs contain both BXI8515 (hip-y/hip-x/knee) and BXI7010 (hip-z).
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_y_joint",
                ".*_hip_x_joint",
                ".*_hip_z_joint",
                ".*_knee_y_joint",
            ],
            effort_limit_sim={
                ".*_hip_y_joint": 150.0,
                ".*_hip_x_joint": 150.0,
                ".*_hip_z_joint": 45.0,
                ".*_knee_y_joint": 150.0,
            },
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_y_joint": STIFFNESS_BXI8515,
                ".*_hip_x_joint": STIFFNESS_BXI8515,
                ".*_hip_z_joint": STIFFNESS_BXI7010,
                ".*_knee_y_joint": STIFFNESS_BXI8515,
            },
            damping={
                ".*_hip_y_joint": DAMPING_BXI8515,
                ".*_hip_x_joint": DAMPING_BXI8515,
                ".*_hip_z_joint": DAMPING_BXI7010,
                ".*_knee_y_joint": DAMPING_BXI8515,
            },
            armature={
                ".*_hip_y_joint": ARMATURE_BXI8515,
                ".*_hip_x_joint": ARMATURE_BXI8515,
                ".*_hip_z_joint": ARMATURE_BXI7010,
                ".*_knee_y_joint": ARMATURE_BXI8515,
            },
        ),

        # Both ankle axes use BXI5018-based parallel mechanisms, but their
        # joint-side torque limits and equivalent inertias are different.
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_y_joint",
                ".*_ankle_x_joint",
            ],
            effort_limit_sim={
                ".*_ankle_y_joint": 40.0,
                ".*_ankle_x_joint": 15.0,
            },
            velocity_limit_sim=20.0,
            stiffness={
                ".*_ankle_y_joint": STIFFNESS_ANKLE_Y,
                ".*_ankle_x_joint": STIFFNESS_ANKLE_X,
            },
            damping={
                ".*_ankle_y_joint": DAMPING_ANKLE_Y,
                ".*_ankle_x_joint": DAMPING_ANKLE_X,
            },
            armature={
                ".*_ankle_y_joint": ARMATURE_ANKLE_Y,
                ".*_ankle_x_joint": ARMATURE_ANKLE_X,
            },
        ),

        # Waist-y / waist-x are BXI7010 parallel mechanisms; waist-z is BXI8515.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_y_joint",
                "waist_x_joint",
                "waist_z_joint",
            ],
            effort_limit_sim={
                "waist_y_joint": 90.0,
                "waist_x_joint": 100.0,
                "waist_z_joint": 150.0,
            },
            velocity_limit_sim=20.0,
            stiffness={
                "waist_y_joint": STIFFNESS_WAIST_Y,
                "waist_x_joint": STIFFNESS_WAIST_X,
                "waist_z_joint": STIFFNESS_BXI8515,
            },
            damping={
                "waist_y_joint": DAMPING_WAIST_Y,
                "waist_x_joint": DAMPING_WAIST_X,
                "waist_z_joint": DAMPING_BXI8515,
            },
            armature={
                "waist_y_joint": ARMATURE_WAIST_Y,
                "waist_x_joint": ARMATURE_WAIST_X,
                "waist_z_joint": ARMATURE_BXI8515,
            },
        ),

        # Shoulder-y/shoulder-x/elbow-y use BXI7010.
        # Shoulder-z and all wrist axes use BXI5014.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y_joint",
                ".*_shoulder_x_joint",
                ".*_shoulder_z_joint",
                ".*_elbow_y_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_y_joint": 45.0,
                ".*_shoulder_x_joint": 45.0,
                ".*_shoulder_z_joint": 21.0,
                ".*_elbow_y_joint": 45.0,
                ".*_wrist_.*_joint": 21.0,
            },
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_y_joint": STIFFNESS_BXI7010,
                ".*_shoulder_x_joint": STIFFNESS_BXI7010,
                ".*_shoulder_z_joint": STIFFNESS_BXI5014,
                ".*_elbow_y_joint": STIFFNESS_BXI7010,
                ".*_wrist_.*_joint": STIFFNESS_BXI5014,
            },
            damping={
                ".*_shoulder_y_joint": DAMPING_BXI7010,
                ".*_shoulder_x_joint": DAMPING_BXI7010,
                ".*_shoulder_z_joint": DAMPING_BXI5014,
                ".*_elbow_y_joint": DAMPING_BXI7010,
                ".*_wrist_.*_joint": DAMPING_BXI5014,
            },
            armature={
                ".*_shoulder_y_joint": ARMATURE_BXI7010,
                ".*_shoulder_x_joint": ARMATURE_BXI7010,
                ".*_shoulder_z_joint": ARMATURE_BXI5014,
                ".*_elbow_y_joint": ARMATURE_BXI7010,
                ".*_wrist_.*_joint": ARMATURE_BXI5014,
            },
        ),

        # ELF3 31-DoF control API uses head_z_joint/head_y_joint. These are
        # BXI5014-class parameters at the joint side.
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_z_joint",
                "head_y_joint",
            ],
            effort_limit_sim=21.0,
            velocity_limit_sim=20.0,
            stiffness=STIFFNESS_BXI5014,
            damping=DAMPING_BXI5014,
            armature=ARMATURE_BXI5014,
        ),
    }
)


ELF3_ACTION_SCALE = {}

for a in ELF3_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)

    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)

    for n in names:
        if n in e and n in s and s[n]:
            ELF3_ACTION_SCALE[n] = 0.25 * e[n] / s[n]