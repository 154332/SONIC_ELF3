# ELF3 静止到行走的运动学参考

输出：csv/Neutral_stand_to_walk_002__A057.csv。
原 CSV、现有 PKL、训练代码和部署代码均未修改。

## 时间和格式

- 120 FPS，1448 帧，Frame 0–1447，约12.058秒，38列。
- 与原输入保持相同列名与顺序：root translation 为厘米，root Euler 和31个关节角度为度。
- 0–2秒：固定站姿，左右 hip_y=-0.15rad、knee_y=0.30rad、ankle_y=-0.15rad，其余关节为0；root roll/pitch=0，yaw沿用原动作初始朝向。相邻帧姿态完全相同，初始差分速度为0。
- 2–4秒：五次 smootherstep 混合站姿与推进中的 walk，同时原动作时间速度以 smoothstep 从0升至1；root quaternion 使用旋转插值，避免直接线性混合欧拉角。
- 4秒时连接原动作的1秒位置，之后继续原 walk 到结尾。原动作最初1秒由2秒过渡重构，不是简单重复首帧后突然接回walk。
- 根据部署 MuJoCo 模型对全部帧做root Z对齐：最低脚底capsule位于z=0，另一脚不穿地。后续walk的root Z也因此与原CSV不同，最大高度调整约4.27cm；后续关节角、root XY和朝向继续沿用原动作。

## 检查结果与限制

初始2秒静止、数值有限、31关节限位、脚底几何和120→30FPS真实转换器检查通过，见validation.json。
这是运动学候选参考，未执行足底固定IK、接触力/重心优化或闭环试验。高度对齐不代表真实接触；接触脚切换时可能产生高度速度变化，仍可能出现滑脚或失稳。不能宣称已得到动力学有效的起步动作或保证重新训练成功。

## 转为独立PKL

在项目根目录、使用已有env_isaaclab环境：

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib_elf3.py \
  --input artifacts/elf3_stand_to_walk/csv \
  --output artifacts/elf3_stand_to_walk/pkl \
  --fps_source 120 --fps 30 --individual
```

本次没有创建PKL。初步起步训练应从该动作第0帧开始，在同一个episode内经过站立和起步；只把它加入随机帧采样目录，可能绕过起步段。
不要把站立段经过后的机器人重置为walk姿态/速度。部署的暂停与blend仍需与训练任务对齐。

独立生成器generate_startup.py可用env_isaaclab中的Python重跑，仅写入本文件夹。它会校验原文件哈希未变化。
