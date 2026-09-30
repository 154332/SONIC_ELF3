# ONNX 010000 与 mode 0 部署配置核对

训练目录：/home/robot/GR00T-WholeBodyControl/logs_rl/TRL_ELF3_Track/manager/universal_token/all_modes/sonic_elf3_test-20260929_204859

训练 config.yaml 中 g1 encoder 使用 motion_anchor_ori_b_mf_nonflat；对应 observation 函数使用完整机器人朝向。现有部署 YAML 使用 heading 特征。

副本仅替换 mode 0 对应的 10 帧朝向特征定义与引用，共两处；teleop、SMPL 未修正。本次未修改现有代码和配置，也未运行闭环验证。

这确认了配置语义差异，不证明它是所有跌倒的唯一原因；完整输入向量逐项对照仍未执行。

训练 motion_file 指向 data/elf3_motions/robot，不能由此还原训练时目录中的确切动作列表。

## ONNX SHA-256

- model_step_010000_decoder.onnx: ec21732a8fb0257356a889044ab6ff58f5ee03eb9fc44698f9089c04e8862a68；与部署文件一致

- model_step_010000_encoder.onnx: 03853c6cbf80457dfe5b14794d9d70d5c24d364f5d359de4328e4d2c1004de5e；与部署文件一致

- model_step_010000_g1.onnx: 2f44d54b4d3b507944b5377db466d10be25c4840bd158093fc85c7c321f7298b；与部署文件一致

- model_step_010000_smpl.onnx: 7216cad6917205e12aef2fa743255db104b7634321ce20ecf319a31e2ed7b877；与部署文件一致

- model_step_010000_teleop.onnx: de8bac63bc13c2548f61f5f541d31ed8733206f79605e6082c097cc87a59b829；与部署文件一致
