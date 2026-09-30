# ELF3 暂停→播放 reference blending 修改说明

两个生产文件均已在原路径完成修改；完整内容随 ZIP 提供，未省略原有函数。
逐行新增/删除/修改见 reference_blend.patch，以此次修改前的本地文件为基准，未将你此前的 ELF3 移植改动包含进去。

## 使用

在仓库根目录执行：

```bash
bash gear_sonic_deploy/deploy_elf3.sh --no-planner --reference-blend-duration 1.0 sim
```

默认就是 1 秒。设为 0 恢复原始直接切换，设为 0.5 或 2 调整时长。
脚本会自动 just build，之后继续原有启动流程。
终端会输出 [ReferenceBlend] Start、alpha（约每 20%）、Complete。

## 实际行为与范围

- 只在本地普通 motion 的 encoder mode 0、使用本地 encoder、无外部 upper-body/VR 数据时启用。
- Planner、streamed、temporary_motion、encoder mode 1/2、外部 token/VR/upper-body 输入保留原逻辑。
- “stand”是按 T 前暂停时使用的 reference frame，并不是强制默认站姿，也不是每 tick 更新的机器人实测姿态。
- 第一帧 alpha=0，1 秒后 alpha=1；一个 tick 内整个 future window 使用同一个 alpha。
- walking 时间轴正常前进，不改变 current_frame_++，不生成新的 pkl，不修改 CSV 或 ONNX。
- q/dq/position 使用独立 cross-fade，quaternion 调用现有 SLERP；dq 是混合的 reference 特征，并非混合 q 的严格时间导数。
- 此次只处理暂停→播放的起步；暂停、motion 切换、模式切换、重置会取消当前混合，不提供 walk→stand 的减速过渡或任意 motion 间的 cross-fade。
- mode 0 默认 observation 不含 body position；辅助读取函数仍支持配置中启用的 body position/root Z。其他模式不会因这些辅助函数而自动获得起步混合。
- UpdateHeadingState 保留原始 motion 的初始化朝向，防止混合权重变化带动 heading 基准漂移。
- 原有 target-motion 日志/录制仍记录源 motion；--policy-input-logfile 记录 decoder observation（包含 token），并不直接记录 encoder 的 q/dq reference。终端 alpha 日志用于确认过渡触发与完成；数值测试直接检查混合后的 reference。

## C++ 新增与修改代码

| 类型 | 代码位置（修改后行号） | 内容 |
|---|---|---|
| 新增 | `IsValidReferenceBlendDuration`，第 76 行 | 验证时长，支持使用 -ffast-math 的 Release 构建，拒绝 NaN/Inf/负值。 |
| 新增 | `状态与冻结快照`，第 236 行 | 保存持续时间、已过时间、alpha、上一周期播放状态和 motion/mode/frame，以及冻结 reference 与预分配混合缓存。 |
| 新增 | `ResetReferenceBlend`，第 454 行 | 清除过渡状态，保留缓存容量。 |
| 新增 | `UpdateReferenceBlend`，第 462 行 | 检测暂停→播放；复制暂停 reference；每 tick 确定统一 alpha；处理暂停/重置/回退/换 motion/换 mode；输出进度。 |
| 新增 | `ReadReferenceJointPositions`，第 535 行 | q_ref = (1-alpha)*q_pause + alpha*q_walk。 |
| 新增 | `ReadReferenceJointVelocities`，第 545 行 | dq_ref = alpha*dq_walk，静态端速度为零。 |
| 新增 | `ReadReferenceBodyPositions`，第 555 行 | 所有已存储 body 的 XYZ 线性混合。 |
| 新增 | `ReadReferenceBodyQuaternions`，第 567 行 | 调用现有 quat_slerp_d，使用 wxyz 顺序。 |
| 修改 | `Root Z / Body position gatherers`，第 578 行 | 从 ReadReferenceBodyPositions 读取。 |
| 修改 | `Anchor orientation / heading diff`，第 767 行 | 先读取混合 quaternion，再应用 heading 和转换到 rotation observation；refheading 首帧也同步。 |
| 修改 | `Joint position / velocity gatherers`，第 891 行 | 全关节与子集分支共享混合后的源数据，包括下肢和手腕；暂停速度置零逻辑保留。 |
| 修改 | `Motion-derived VR3 / VR5 gatherers`，第 1257 行 | 统一读取 body position/quaternion 辅助函数；外部 VR 数据仍按原逻辑处理。 |
| 修改 | `ELF3Deploy 构造函数`，第 2304 行 | 增加默认值 1.0 的最后一个参数，验证并保存配置。 |
| 修改 | `Control`，第 4107 行 | 持锁且在 GatherObservations 前更新一次 alpha，成功采集后按 control_dt_ 累计时间。 |
| 修改 | `main 参数解析与帮助`，第 4549 行 | 支持 --reference-blend-duration；完整解析数值并校验，再传给构造函数。 |

## 启动脚本新增与修改代码

- 帮助与示例新增 --reference-blend-duration SECONDS。
- 新增 REFERENCE_BLEND_DURATION="1.0" 默认配置。
- case 参数解析新增时长分支，拒绝缺值、负数和非数字格式。
- 启动配置输出新增 Reference blend 时长。
- RUN_ARGS 新增 --reference-blend-duration 与对应数值，传入 C++ 程序。

## 验证

- `cmake --build gear_sonic_deploy/build --target elf3_deploy_onnx_ref -j 2`：编译、链接通过。
- `bash -n gear_sonic_deploy/deploy_elf3.sh`：通过。
- 启动脚本 --help、缺值/非法值检查：通过。
- 模拟 build/run 的启动检查：0、0.5、1.0、2、1e-1 时长参数和仿真 CRC 参数正确传递。
- 实际 Release 可执行文件在 DDS 初始化前拒绝 nan/NaN/inf/-inf/-1/1junk/1e9999 和缺失参数。
- `python3 gear_sonic_deploy/reference_blend_review/check_reference_blend.py`：直接提取此次 C++ 中的真实函数，以合成 motion 验证 0.5/1/2 秒过渡、整个 future window、关节子集、位置、速度、body、四元数单位长度与短路径、重复 T、暂停后重启、重置/换 motion/换 mode、禁用和排除路径。
- 数值检查启用 AddressSanitizer/UndefinedBehaviorSanitizer；当前桌面沙箱使用 ptrace，测试禁用不兼容的 LeakSanitizer。
- 未运行 MuJoCo 起步成功率实验，不能据此声称机器人已能稳定起步。

## 文件大小与差异

- gear_sonic_deploy/deploy_elf3.sh: 353 lines, +16/-1
- gear_sonic_deploy/src/elf3/elf3_deploy_onnx_ref/src/elf3_deploy_onnx_ref.cpp: 4711 lines, +207/-21
