import ast
import mujoco


XML_PATH = "/home/robot/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/elf3/elf3.xml"
ELF3_CONFIG_PATH = "/home/robot/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/elf3.py"
OUTPUT_FILE = "elf3_mapping.py"


def load_isaaclab_joints():
    with open(ELF3_CONFIG_PATH, "r") as f:
        tree = ast.parse(f.read())

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "ELF3_ISAACLAB_JOINTS":
                    return ast.literal_eval(node.value)

    raise RuntimeError("Cannot find ELF3_ISAACLAB_JOINTS")


def get_mujoco_joints(model):
    joints = []

    for i in range(model.njnt):
        name = model.joint(i).name

        if name != "world_joint":
            joints.append(name)

    return joints


def get_mujoco_bodies(model):
    bodies = []

    for i in range(model.nbody):
        bodies.append(model.body(i).name)

    return bodies


def build_mapping():

    model = mujoco.MjModel.from_xml_path(XML_PATH)

    mujoco_joints = get_mujoco_joints(model)
    mujoco_bodies = get_mujoco_bodies(model)

    isaaclab_joints = load_isaaclab_joints()

    isaaclab_to_mujoco_dof = []

    for joint in isaaclab_joints:
        if joint not in mujoco_joints:
            raise RuntimeError(
                f"{joint} not found in MuJoCo"
            )

        isaaclab_to_mujoco_dof.append(
            mujoco_joints.index(joint)
        )

    mujoco_to_isaaclab_dof = [0] * len(
        isaaclab_to_mujoco_dof
    )

    for isaac_id, mujoco_id in enumerate(
        isaaclab_to_mujoco_dof
    ):
        mujoco_to_isaaclab_dof[mujoco_id] = isaac_id


    with open(OUTPUT_FILE, "w") as f:
        f.write("ELF3_ISAACLAB_TO_MUJOCO_MAPPING = {\n")
        f.write(f"    'isaaclab_joints': {isaaclab_joints},\n")
        f.write(f"    'isaaclab_to_mujoco_dof': {isaaclab_to_mujoco_dof},\n")
        f.write(f"    'mujoco_to_isaaclab_dof': {mujoco_to_isaaclab_dof},\n")
        f.write("}\n")


    print(f"Generated: {OUTPUT_FILE}")
    print(f"IsaacLab DOF: {len(isaaclab_joints)}")
    print(f"MuJoCo DOF: {len(mujoco_joints)}")
    print(f"IsaacLab -> MuJoCo: {isaaclab_to_mujoco_dof}")


if __name__ == "__main__":
    build_mapping()