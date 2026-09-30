# ELF3 相同参考姿态、零初始速度诊断

在用于 IsaacLab 评估的 Python 环境中，从项目根目录依次运行：

```bash
bash artifacts/elf3_zero_velocity_eval/run_eval.sh normal
bash artifacts/elf3_zero_velocity_eval/run_eval.sh zero
```

两组使用相同 PT checkpoint、单条 walk PKL、seed、mode g1、初始帧和现有评估入口。
normal 使用项目原始 TrackingCommand；zero 在评估重置时先执行原始初始化，再保留实际关节位置与 root pose，将实际关节速度、root 线速度和角速度清零。
参考 PKL、reference q/dq、未来窗口、模型权重均不改变。zero 仅在 is_evaluating=True 时生效，终端应出现 [ZeroVelocityEval]。

这个测试仍从 walk 首帧姿态开始，不等于“站立两秒再按 T”，没有部署 blend 和拉力。它用于隔离初始速度这一因素；若失败，不能单独证明连续起步训练是唯一原因。

两组都沿用项目评估对 train-only events 的处理。不是所有 startup 物理随机化都自动移除；保持相同 seed/参数对照，未声称完全关闭所有随机化。

原有代码没有修改。新增 Python 通过语法检查，shell 通过 bash -n，调用 API 与本机 IsaacLab 源码核对过；未启动 GPU 闭环试验，完整运行仍需实际验证。

## 评估统计兼容修复

原 im_eval_callback 写死 G1 body 名称，在 ELF3 提前终止后的结果汇总阶段抛出 ValueError。独立副本 im_eval_callback_elf3.py 将统计用 body 子集对应到 ELF3（pelvis 对应 waist_z_link）。这只影响指标分组，不改变策略、奖励或终止阈值。run_eval.sh 的两组均使用该副本。提前终止打印 [ELF3Eval] 和 extras 中的 termination 统计；统计项不一定只代表当前单个环境，须结合单环境测试解释。
