/**
 * @file robot_parameters.hpp
 * @brief ELF3 low-level deployment constants and joint indices.
 *
 * Joint indices in this file use the ELF3 MuJoCo/actuator order used by the
 * sim2sim bridge. Real-hardware deployment must replace/verify this mapping
 * against the ELF3 hardware motor IDs before sending commands to the robot.
 */
#ifndef ROBOT_PARAMETERS_HPP
#define ROBOT_PARAMETERS_HPP
#include <array>
#include <string>
static const std::string HG_CMD_TOPIC="rt/lowcmd";
static const std::string HG_IMU_TORSO="rt/secondary_imu";
static const std::string HG_STATE_TOPIC="rt/lowstate";
constexpr int ELF3_NUM_MOTOR=31;
constexpr bool ELF3_HAS_DEX3_HANDS=false;
struct MotorCommand {
    std::array<float,ELF3_NUM_MOTOR> q_target={};
    std::array<float,ELF3_NUM_MOTOR> dq_target={};
    std::array<float,ELF3_NUM_MOTOR> kp={};
    std::array<float,ELF3_NUM_MOTOR> kd={};
    std::array<float,ELF3_NUM_MOTOR> tau_ff={};
};
struct HeadingState {
    std::array<double,4> init_base_quat;
    double delta_heading;
    HeadingState(const std::array<double,4>& quat={1.0,0.0,0.0,0.0},double delta=0.0):init_base_quat(quat),delta_heading(delta) {}
};
struct OperatorState { bool stop=false; bool start=false; bool play=false; };
// LowCmd still exposes mode_pr; ELF3 sim2sim keeps the field at PR=0.
enum class Mode { PR=0,AB=1 };
enum ELF3JointIndex {
    WaistY=0,WaistX=1,WaistZ=2,
    LeftHipY=3,LeftHipX=4,LeftHipZ=5,LeftKneeY=6,LeftAnkleY=7,LeftAnkleX=8,
    RightHipY=9,RightHipX=10,RightHipZ=11,RightKneeY=12,RightAnkleY=13,RightAnkleX=14,
    LeftShoulderY=15,LeftShoulderX=16,LeftShoulderZ=17,LeftElbowY=18,LeftWristX=19,LeftWristY=20,LeftWristZ=21,
    RightShoulderY=22,RightShoulderX=23,RightShoulderZ=24,RightElbowY=25,RightWristX=26,RightWristY=27,RightWristZ=28,
    HeadZ=29,HeadY=30
};
#endif
