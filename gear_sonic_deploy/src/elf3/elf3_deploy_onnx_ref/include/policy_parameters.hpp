/**
 * @file policy_parameters.hpp
 * @brief ELF3 policy-side joint ordering, gains, action scales and default pose.
 *
 * Keep this file synchronized with:
 *   gear_sonic/envs/manager_env/robots/elf3.py
 *
 * Training uses:
 *   Kp = armature * NATURAL_FREQ^2
 *   Kd = 2 * DAMPING_RATIO * armature * NATURAL_FREQ
 *   action_scale = 0.25 * effort_limit_sim / Kp
 *
 * All per-joint arrays below are in ELF3 MuJoCo/controller order:
 *   waist(3), left leg(6), right leg(6), left arm(7), right arm(7), head(2)
 */
#ifndef POLICY_PARAMETERS_HPP
#define POLICY_PARAMETERS_HPP

#include <array>
#include <vector>
#include "robot_parameters.hpp"

constexpr double ONE_DEGREE = 0.0174533;
constexpr double NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535;
constexpr double DAMPING_RATIO = 2.0;

// Joint-side equivalent inertias [kg*m^2], matching elf3.py.
constexpr double ARMATURE_BXI8515 = 0.044688;
constexpr double ARMATURE_BXI7010 = 0.0137351;
constexpr double ARMATURE_BXI5014 = 0.00424198;
constexpr double ARMATURE_WAIST_Y = 0.0274702;
constexpr double ARMATURE_WAIST_X = 0.0412054;
constexpr double ARMATURE_ANKLE_Y = 0.00848397;
constexpr double ARMATURE_ANKLE_X = 0.00551458;

constexpr double stiffness(double armature) {
  return armature * NATURAL_FREQ * NATURAL_FREQ;
}
constexpr double damping(double armature) {
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ;
}
constexpr double action_scale(double effort, double kp) {
  return 0.25 * effort / kp;
}

constexpr double STIFFNESS_BXI8515 = stiffness(ARMATURE_BXI8515);
constexpr double STIFFNESS_BXI7010 = stiffness(ARMATURE_BXI7010);
constexpr double STIFFNESS_BXI5014 = stiffness(ARMATURE_BXI5014);
constexpr double STIFFNESS_WAIST_Y = stiffness(ARMATURE_WAIST_Y);
constexpr double STIFFNESS_WAIST_X = stiffness(ARMATURE_WAIST_X);
constexpr double STIFFNESS_ANKLE_Y = stiffness(ARMATURE_ANKLE_Y);
constexpr double STIFFNESS_ANKLE_X = stiffness(ARMATURE_ANKLE_X);

constexpr double DAMPING_BXI8515 = damping(ARMATURE_BXI8515);
constexpr double DAMPING_BXI7010 = damping(ARMATURE_BXI7010);
constexpr double DAMPING_BXI5014 = damping(ARMATURE_BXI5014);
constexpr double DAMPING_WAIST_Y = damping(ARMATURE_WAIST_Y);
constexpr double DAMPING_WAIST_X = damping(ARMATURE_WAIST_X);
constexpr double DAMPING_ANKLE_Y = damping(ARMATURE_ANKLE_Y);
constexpr double DAMPING_ANKLE_X = damping(ARMATURE_ANKLE_X);

// IsaacLab body indices: left wrist, right wrist, torso, left ankle, right ankle.
const std::array<int, 5> vr_5point_index = {24, 25, 0, 30, 31};
const std::array<int, 3> vr_3point_index = {24, 25, 0};

// The external SONIC upper-body streaming interface intentionally remains 17 DoF:
// waist(3) + left arm(7) + right arm(7), excluding head_z/head_y.
// This is independent from the 31-DoF policy output. Change it only if the
// external ZMQ/ROS/VLA protocol is also upgraded to a 19-DoF upper-body payload.
const std::vector<int> upper_body_joint_mujoco_order_in_isaaclab_index =
    {3, 7, 10, 1, 5, 8, 11, 15, 19, 23, 2, 6, 9, 12, 16, 20, 24};
const std::vector<int> upper_body_joint_mujoco_order_in_mujoco_index =
    {0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
const std::vector<int> upper_body_joint_isaaclab_order_in_isaaclab_index =
    {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 19, 20, 23, 24};
const std::vector<int> upper_body_joint_isaaclab_order_in_mujoco_index =
    {15, 22, 0, 16, 23, 1, 17, 24, 2, 18, 25, 19, 26, 20, 27, 21, 28};

// ELF3 wrists in IsaacLab order: L/R wrist_x, L/R wrist_y, L/R wrist_z.
const std::vector<int> wrist_joint_mujoco_order_in_isaaclab_index = {15, 19, 23, 16, 20, 24};
const std::vector<int> wrist_joint_mujoco_order_in_mujoco_index = {19, 20, 21, 26, 27, 28};
const std::vector<int> wrist_joint_isaaclab_order_in_isaaclab_index = {15, 16, 19, 20, 23, 24};
const std::vector<int> wrist_joint_isaaclab_order_in_mujoco_index = {19, 26, 20, 27, 21, 28};

const std::vector<int> lower_body_joint_mujoco_order_in_isaaclab_index =
    {13, 17, 21, 25, 27, 29, 14, 18, 22, 26, 28, 30};
const std::vector<int> lower_body_joint_mujoco_order_in_mujoco_index =
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
const std::vector<int> lower_body_joint_isaaclab_order_in_isaaclab_index =
    {13, 14, 17, 18, 21, 22, 25, 26, 27, 28, 29, 30};
const std::vector<int> lower_body_joint_isaaclab_order_in_mujoco_index =
    {3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14};

// output_mujoco[i] = input_isaaclab[isaaclab_to_mujoco[i]].
const std::array<int, ELF3_NUM_MOTOR> isaaclab_to_mujoco = {
    3, 7, 10, 13, 17, 21, 25, 27, 29, 14, 18, 22, 26, 28, 30,
    1, 5, 8, 11, 15, 19, 23, 2, 6, 9, 12, 16, 20, 24, 0, 4};

// output_isaaclab[i] = input_mujoco[mujoco_to_isaaclab[i]].
const std::array<int, ELF3_NUM_MOTOR> mujoco_to_isaaclab = {
    29, 15, 22, 0, 30, 16, 23, 1, 17, 24, 2, 18, 25, 3, 9,
    19, 26, 4, 10, 20, 27, 5, 11, 21, 28, 6, 12, 7, 13, 8, 14};

// Effort limits [Nm] in MuJoCo/controller order, matching elf3.py.
const std::array<double, ELF3_NUM_MOTOR> effort_limits = {
    90.0, 100.0, 150.0,
    150.0, 150.0, 45.0, 150.0, 40.0, 15.0,
    150.0, 150.0, 45.0, 150.0, 40.0, 15.0,
    45.0, 45.0, 21.0, 45.0, 21.0, 21.0, 21.0,
    45.0, 45.0, 21.0, 45.0, 21.0, 21.0, 21.0,
    21.0, 21.0};

// Kp in MuJoCo/controller order.
const std::array<float, ELF3_NUM_MOTOR> kps = {
    static_cast<float>(STIFFNESS_WAIST_Y),
    static_cast<float>(STIFFNESS_WAIST_X),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_ANKLE_Y),
    static_cast<float>(STIFFNESS_ANKLE_X),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI8515),
    static_cast<float>(STIFFNESS_ANKLE_Y),
    static_cast<float>(STIFFNESS_ANKLE_X),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI7010),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI5014),
    static_cast<float>(STIFFNESS_BXI5014)};

// Kd in MuJoCo/controller order.
const std::array<float, ELF3_NUM_MOTOR> kds = {
    static_cast<float>(DAMPING_WAIST_Y),
    static_cast<float>(DAMPING_WAIST_X),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_ANKLE_Y),
    static_cast<float>(DAMPING_ANKLE_X),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI8515),
    static_cast<float>(DAMPING_ANKLE_Y),
    static_cast<float>(DAMPING_ANKLE_X),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI7010),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI5014),
    static_cast<float>(DAMPING_BXI5014)};

// Per-joint policy action scales, matching elf3.py's
// ELF3_ACTION_SCALE[n] = 0.25 * effort_limit_sim[n] / stiffness[n].
const std::array<double, ELF3_NUM_MOTOR> elf3_action_scale = {
    action_scale(90.0, STIFFNESS_WAIST_Y),
    action_scale(100.0, STIFFNESS_WAIST_X),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(40.0, STIFFNESS_ANKLE_Y),
    action_scale(15.0, STIFFNESS_ANKLE_X),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(150.0, STIFFNESS_BXI8515),
    action_scale(40.0, STIFFNESS_ANKLE_Y),
    action_scale(15.0, STIFFNESS_ANKLE_X),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(45.0, STIFFNESS_BXI7010),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(21.0, STIFFNESS_BXI5014),
    action_scale(21.0, STIFFNESS_BXI5014)};

const std::array<double, ELF3_NUM_MOTOR> default_angles = {
    0.0, 0.0, 0.0,

    -0.15, 0.0, 0.0, 0.30, -0.15, 0.0,

    -0.15, 0.0, 0.0, 0.30, -0.15, 0.0,

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    0.0, 0.0
};

#endif  // POLICY_PARAMETERS_HPP
