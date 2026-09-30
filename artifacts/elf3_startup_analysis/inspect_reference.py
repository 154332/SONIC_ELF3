import ast,csv,json,re
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
ref=root/'gear_sonic_deploy/reference/elf3/Neutral_walk_forward_002__A057'
def read(name):
 with (ref/(name+'.csv')).open() as f:
  rows=csv.reader(f);next(rows);return np.array([[float(v) for v in r] for r in rows])
q,dq,p,quat,vel=[read(n) for n in ['joint_pos','joint_vel','body_pos','body_quat','body_lin_vel']]
policy=(root/'gear_sonic_deploy/src/elf3/elf3_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
maptext=re.search(r'isaaclab_to_mujoco\s*=\s*\{([^}]+)\}',policy).group(1)
order=[int(v) for v in re.findall(r'\d+',maptext)]
model=mujoco.MjModel.from_xml_path(str(root/'gear_sonic_deploy/elf3/mjcf/elf3.xml'))
data=mujoco.MjData(model)
frames=[]
for f in [0,5,10,25,50,75,100]:
 data.qpos[:3]=p[f,:3];data.qpos[3:7]=quat[f,:4];data.qpos[7:]=q[f,order]
 mujoco.mj_forward(model,data)
 foot={}
 for side in ['l','r']:
  ids=[i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(side+'_foot')]
  foot[side]=min(float(data.geom_xpos[i,2]-model.geom_size[i,0]-model.geom_size[i,1]*abs(data.geom_xmat[i].reshape(3,3)[2,2])) for i in ids)
 frames.append({'frame':f,'seconds':f/50,'joint_velocity_rms_rad_s':float(np.sqrt(np.mean(dq[f]**2))),'root_horizontal_displacement_m':float(np.linalg.norm(p[f,:2]-p[0,:2])),'foot_collision_lowest_z_m':foot})
report={'reference':str(ref),'robot_mass_kg':float(model.body_mass.sum()),'first_root_velocity_m_s':vel[0,:3].tolist(),'first_joint_velocity_maxabs_rad_s':float(np.max(abs(dq[0]))),'first_root_roll_pitch_yaw_deg':Rotation.from_quat(quat[0,[1,2,3,0]]).as_euler('xyz',degrees=True).tolist(),'sampled_frames':frames,'limitations':'FK of stored reference only; not measured running robot contacts or controller stability.'}
(folder/'reference_numbers.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
