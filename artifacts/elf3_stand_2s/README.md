# ELF3 2 秒静止站立 CSV

文件：`Neutral_stand_2s_elf3.csv`。本次只在当前文件夹创建文件，没有覆盖已有代码或 motion。

- 120 FPS，241 帧（Frame 0–240），包含 t=0 和 t=2 秒两个端点。
- 格式：Frame + root XYZ + root Euler XYZ + 31 个 ELF3 关节，共 38 列。
- root 平移单位 cm，root 欧拉角与关节角单位 degree，匹配现有 ELF3 CSV→PKL 转换器。
- 所有姿态帧完全相同，只有 Frame 递增；由姿态差分得到的关节/root 速度均为 0。
- 左右髋 pitch=-0.15 rad，膝=0.30 rad，踝 pitch=-0.15 rad；其余关节为 0。
- root X/Y=0，roll/pitch/yaw=0，root Z=105.531348987907 cm。
- 使用当前训练 MJCF 做 FK，将默认 1.08 m root 高度下降约 2.47 cm，使双脚 capsule 碰撞几何最低点位于 z=0。
- 已检查关节限位、左右脚底高度、COM 投影位于双脚联合边界框中。该几何检查不是动态稳定性实验。
- 已调用现有转换器的真实 convert_motion() 验证，120→30 FPS 得到 61 帧、dof=(61,31)、pose_aa=(61,32,3)，姿态差分为 0。

在仓库根目录，使用装有 pandas/scipy/joblib 的现有环境转换：

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib_elf3.py \
  --input artifacts/elf3_stand_2s \
  --output artifacts/elf3_stand_2s/pkl \
  --fps_source 120 \
  --fps 30 \
  --individual
```

输出为 `artifacts/elf3_stand_2s/pkl/Neutral_stand_2s_elf3.pkl`。
本次按请求只提供 CSV，未生成或替换训练 PKL，也未修改训练配置。

加入独立 stand motion 提供静态站立训练目标，但不会自动使训练包含“静止→walk”的连续过渡；仍应核实训练采样、起步初始化以及训练/部署 observation 一致性。

你提供的 `data/elf3_motions/robot/Neutral_walk_forward_002__A057.csv` 在本次工作区不存在，因此没有直接验证该原始 CSV 或同名 PKL 的首帧。之前测得的非零首帧速度来自 `gear_sonic_deploy/reference/elf3/Neutral_walk_forward_002__A057/` 的部署 reference。

`generate_stand_csv.py` 是放在本文件夹中的独立生成脚本，需要 numpy 和 mujoco；生成时只读取现有转换器和 MJCF，不修改它们。`validation.json` 记录本次检查结果。
