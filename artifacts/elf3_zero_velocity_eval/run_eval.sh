#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
MODE="${1:-}"
if [[ "$MODE" != normal && "$MODE" != zero ]]; then
    echo "Usage: bash artifacts/elf3_zero_velocity_eval/run_eval.sh normal|zero [Hydra overrides...]"
    exit 2
fi
shift
TASK_ARGS=()
if [[ "$MODE" == zero ]]; then
    TASK_ARGS+=("++manager_env.commands.motion._target_=artifacts.elf3_zero_velocity_eval.zero_velocity_command.ZeroVelocityTrackingCommandCfg")
fi
python gear_sonic/eval_agent_trl.py \
    "++checkpoint=$PROJECT_DIR/logs_rl/TRL_ELF3_Track/manager/universal_token/all_modes/sonic_elf3_test-20260929_204859/model_step_010000.pt" \
    ++headless=False \
    ++num_envs=1 \
    ++seed=0 \
    ++use_encoder=g1 \
    ++eval_callbacks=im_eval \
    ++callbacks.im_eval._target_=artifacts.elf3_zero_velocity_eval.im_eval_callback_elf3.ImEvalCallback \
    ++run_eval_loop=False \
    "++eval_name=elf3_${MODE}_initial_velocity" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PROJECT_DIR/data/elf3_motions/robot/Neutral_walk_forward_002__A057.pkl" \
    ++manager_env.commands.motion.start_from_first_frame=True \
    ++manager_env.commands.motion.sample_from_n_initial_frames=null \
    "${TASK_ARGS[@]}" "$@"
