import mujoco


XML_PATH = "/home/robot/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/elf3/elf3.xml"


model = mujoco.MjModel.from_xml_path(XML_PATH)

print("========== MuJoCo DOF ==========")

mujoco_joints = []

for i in range(model.njnt):
    name = model.joint(i).name

    if name == "world_joint":
        continue

    mujoco_joints.append(name)
    print(i, name)


print("\n========== MuJoCo BODY ==========")

mujoco_bodies = []

for i in range(model.nbody):
    name = model.body(i).name
    mujoco_bodies.append(name)
    print(i, name)