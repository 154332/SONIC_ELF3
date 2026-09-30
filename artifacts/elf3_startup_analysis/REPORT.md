# ELF3 松开 ElasticBand 后按 T 起步失败：本地代码审计

本次仅新增 artifacts/elf3_startup_analysis 内的审计文件，未修改现有训练/部署/仿真代码。

## 判断

目前有足够源码证据说明，T 触发的是本地 reference 播放；不存在 T 后等待 VLA 生成 trajectory 的流程。
现象最符合“静态支撑→动态步态的状态不匹配，加上外部六维稳定器撤除”的起步问题。
不能仅凭源码给训练或部署分配故障百分比；需要实际 checkpoint 的 resolved config 和起步日志确认。
应先核实 observation 一致性和 reference，再判断是否需要起步训练。不能先断言加一个 stand PKL 或下载大量 SMPL 就会修复。

## 已确认的源码事实

1. ElasticBand 是六维外部稳定器。
   - gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:366–410。
   - point=(0,0,1.075)，kp_pos=10000 N/m，kd_pos=1000 N·s/m，kp_ang=1000 N·m/rad，kd_ang=10 N·m·s/rad。
   - F=kp*(point-position)+kd*(0-linear_velocity)，并施加使世界朝向趋于单位旋转的姿态力矩。
   - 1 cm 位置偏差对应 100 N 弹性项；1° 旋转误差对应约 17.45 N·m 姿态项。
   - 当前 MuJoCo 模型质量约43.22 kg，重力约424 N，100 N不是可忽略的小辅助力。
   - base_sim.py:358–365 将结果写到 torso 的 xfrc_applied；禁用后直接清零，没有逐渐释放。
   - 含义：带 band 的“站稳”不证明 policy 自身能站稳；先 T 可让腿开始跟踪并经过起步瞬态，而 band 仍帮忙约束 torso。这是对现象的机制解释，尚未用运行日志验证。

2. T 是启动当前 motion，没有状态匹配或支撑准备阶段。
   - keyboard_handler.hpp:408–419 在锁内设置 operator_state.play=true。
   - GatherMotionJointPositionsMultiFrame / Velocities / AnchorOrientation 根据 play 构造 future reference。
   - 暂停时保持当前 reference frame，速度为0；播放时按 current_frame+i*step 取未来帧。
   - current_frame 正常每50Hz控制周期增加；不会等重心/足底接触就绪。
   - Encoder/Decoder 在 Control 状态每周期已在运行，T 不重新加载模型；CaptureGraph发生在初始化。
   - 历史窗口10帧约0.2秒，并无一个专门“等2秒预测”的状态。

3. 当前 walking 首帧不是静止站立 reference。
   - 检查来源是 gear_sonic_deploy/reference/elf3/Neutral_walk_forward_002__A057，非缺失的原始 CSV/PKL。
   - 首帧关节速度RMS=1.73536 rad/s，最大绝对值6.87306 rad/s。
   - root速度=(0.06069,-0.29263,-0.15149)m/s。
   - root欧拉角roll≈-1.92°、pitch≈9.00°、yaw≈-110.02°。yaw通常会被部署heading对齐，不应直接认作朝向故障。
   - 前0.5秒/1秒的水平reference位移约0.324/0.793m。
   - 使用当前MuJoCo模型、C++实际关节映射、原始root pose进行FK，首帧脚底capsule最低点左0.04576m、右0.05582m（以z=0为地面）。
   - 这是reference几何检查，不是实际机器人接触日志。训练物理资产来自URDF，也应另查训练加载后的高度修正/碰撞几何。
   - mode0默认encoder不含root position，因此绝对脚底高度不会直接作为mode0输入；它仍是动作数据物理可行性和训练初始化需要核实的证据。

4. 当前配置中存在朝向输入语义差异。
   - sonic_elf3.yaml:164使用motion_anchor_ori_b_mf_nonflat。
   - Python observations.py:1022使用机器人anchor完整朝向，包括roll/pitch。
   - deploy observation_config.yaml:51/86使用motion_anchor_orientation_heading_10frame_step5。
   - C++ GatherMotionAnchorOrientationMutiFrame的orientation_mode=1仅使用机器人heading作为相对旋转左端。
   - 两者都是60维，却不是相同输入；机器人roll/pitch偏离时差异增大。
   - band将姿态误差压小，可能掩盖这一差异；撤掉band后差异可能变大。
   - decoder仍有gravity和angular velocity，不是所有倾斜反馈都丢失。
   - 若ONNX由仓库现有sonic_elf3配置导出，这属于部署兼容问题；若模型实际使用heading配置训练，当前部署可能正确。
   - 检查model_step_010000*.onnx：没有metadata保存训练语义；输入尺寸encoder1791、decoder1054、输出action31正确，只能确认维度，不能确认语义。
   - 不应在未知checkpoint配置下直接改成full-orientation。

5. 训练默认初始化任务不同于部署静止起步。
   - commands.py:2841 _resample_command选择motion与起始时间。
   - :3003–3006读取reference root姿态/速度，:3039附近读取joint pose/velocity。
   - :3071–3084将joint pose/velocity和root pose/velocity写入物理仿真，训练可叠加randomization。
   - :3219 _update_command在motion结束时调用上述resample。
   - 因此，随机采样walk中的帧通常让物理状态带着对应运动速度开始，不等同于“站稳后只切reference”。
   - 当前配置freeze_frame_aug=true不是连续stand→walk起步训练的证据；loader中的冻结增强是将某帧及后续保持，并不能自动证明模型学会恢复播放。
   - 若只训练一条walk、且没有连续起步episode，起步鲁棒性不足是合理怀疑；尚无该ONNX的实际训练数据列表、resolved config和评估记录，不能确认。

6. 当前1秒blend不生成动力学起步过程。
   - UpdateReferenceBlend及ReadReference*在encoder参考层cross-fade；站立端冻结为暂停motion帧，不是独立stand姿态。
   - 起步时alpha从0增加，整个future窗口共用同一alpha；walk cursor继续推进。
   - 0.5秒时walk已到frame25但参考动作只混合50%；第一段步幅/速度可能被压小。
   - alpha混合不保证足底接触、支撑多边形、COM速度连续；joint velocity为独立cross-fade，不是混合joint position的严格时间导数。
   - 这可以减小输入突变，也会引入过渡时间；不能据此承诺立即walk或百分百成功。

## 检查过而暂未发现明确错误的部分

- verify_elf3_mapping.py：ALL PASS，关节/body映射完整且可逆。另有elf3_mapping.py中的旧identity表不用于当前C++路径，不以它判断当前部署映射。
- C++默认关节姿态、Kp/Kd、action_scale公式与当前elf3.py相符。
- Python simulator通过当前DDS motor command的q/kp/kd计算PD；WBC YAML里的DEFAULT_MOTOR_ANGLES旧值不直接用于此PD路径。
- 控制名义周期：训练0.005*4=0.02s，部署0.02s；future spacing训练0.1s、部署5*0.02=0.1s。
- PolicyCfg中已声明base_ang_vel/joint_pos/joint_vel/actions/gravity_dir顺序，部署对应顺序一致；不能仅凭YAML defaults中gravity先出现就认定decoder输入顺序错。
- FREE_BASE:false不表示锁死root；代码检查FLOATING_BASE_JOINT=world_joint识别free joint。
- 以上为当前配置/源码检查，不代表历史checkpoint使用相同参数或真实运行没有时序异常。

## 仍需运行证据排除的因素

- 实际terminal1/terminal2命令、加载的checkpoint prefix与observation YAML。
- ONNX对应resolved training config、训练motion列表和从零训练/微调情况。
- 按T前后torso roll/pitch/angular velocity、joint q/dq、encoder reference、motor targets/torque、左右足底接触力与band外力。
- ReferenceBlend Start/alpha/Complete是否出现，实际encoder mode是否0。
- MuJoCo simulated-time/wall-time比与C++周期。模拟器每步sleep补足0.005s，但超时不补步；C++按墙钟推进reference。若仿真跑不满实时速度，会出现reference比机器人快的问题，当前没有运行时记录证实。
- 脚底reference高度是数据原始问题、地面约定还是训练loader修正差异；原始CSV/PKL在当前工作区未找到。

## 优先验证顺序

1. 取得当前checkpoint实际配置，先确认full-orientation/heading、anchor body、joint顺序、feature layout一致。
2. 固定模型和reference，用带band/不带band两类条件记录起步前后相同状态字段；将“释放band时刻”与“T时刻”分别记录。
3. 检查reference首帧和训练加载后的FK/接触，判断是否适合作为零速度暂停姿态。
4. 比较静止起步与按motion姿态/速度初始化后跟踪，后一项只用于诊断，不能作为最终部署修复。
5. 若兼容性核实后静止起步仍差，再增加同一episode内站立→切reference→walk的训练；切换时不得重置机器人到walk速度。
6. 之后才评估blend时长/相位启动/拉力渐退。改变外力释放方式可以帮助诊断，但不能证明policy已独立学会起步。

本次未运行机器人闭环试验，没有改变任何模型、部署配置或仿真控制代码。数值证据见reference_numbers.json，可用inspect_reference.py重算。
