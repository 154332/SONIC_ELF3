#!/usr/bin/env python3
"""Compile actual reference-blend methods with synthetic motions; no GPU/DDS needed."""
from pathlib import Path
import subprocess
import tempfile
import os

root = Path(__file__).resolve().parents[2]
source = root / 'gear_sonic_deploy/src/elf3/elf3_deploy_onnx_ref/src/elf3_deploy_onnx_ref.cpp'
text = source.read_text()

def method(name):
    start = text.rfind('\n', 0, text.index(name + '(')) + 1
    opening = text.index(') {', start) + 2
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]

members = text[text.index('    double reference_blend_duration_'):text.index('    // =========================================================================\n    // Local motion planner')]
methods = '\n'.join(method(n) for n in [
    'ResetReferenceBlend', 'UpdateReferenceBlend', 'ReadReferenceJointPositions',
    'ReadReferenceJointVelocities', 'ReadReferenceBodyPositions',
    'ReadReferenceBodyQuaternions', 'GatherMotionJointPositionsMultiFrame',
    'GatherMotionJointVelocitiesMultiFrame', 'GatherMotionBodyPositionsMultiFrame',
    'GatherMotionRootZPositionMultiFrame'])
preamble = r'''
#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>
constexpr int ELF3_NUM_MOTOR = 31;
const std::array<int, 17> upper_body_joint_isaaclab_order_in_isaaclab_index{};
struct MotionSequence {
  using Point = std::array<double,3>;
  using Quaternion = std::array<double,4>;
  std::string name = "walk";
  int timesteps = 100, mode = 0, joints = 31;
  std::vector<double> q, dq;
  std::vector<Point> p;
  std::vector<Quaternion> quat;
  std::vector<int> indexes{0, 1};
  MotionSequence(): q(3100), dq(3100), p(200), quat(200) {
    for(int f=0;f<100;++f) {
      for(int j=0;j<31;++j) { q[f*31+j]=j+0.01*f; dq[f*31+j]=0.5+j; }
      for(int b=0;b<2;++b) {
        p[f*2+b]={0.01*f+b,0.02*f,1.0+0.001*f};
        quat[f*2+b]={std::cos(0.005*f),0,0,std::sin(0.005*f)};
      }
    }
  }
  int GetNumJoints() const { return joints; }
  int GetNumBodies() const { return 2; }
  int GetNumBodyQuaternions() const { return 2; }
  int GetEncodeMode() const { return mode; }
  const std::vector<int>& BodyPartIndexes() const { return indexes; }
  const double* JointPositions(int f) const { return q.data()+31*f; }
  const double* JointVelocities(int f) const { return dq.data()+31*f; }
  const Point* BodyPositions(int f) const { return p.data()+2*f; }
  const Quaternion* BodyQuaternions(int f) const { return quat.data()+2*f; }
};
class Harness {
public:
  struct { bool play=false; } operator_state;
  std::shared_ptr<const MotionSequence> current_motion_ = std::make_shared<MotionSequence>();
  std::shared_ptr<MotionSequence> planner_motion_;
  int current_frame_=0, saved_frame_for_observation_window_=0;
  bool is_using_encoder_=true, reinitialize_heading_=false;
  bool has_upper_body_data_=false, has_vr_3point_data_=false, has_vr_5point_data_=false;
  std::array<double,17> upper_body_joint_positions_buffer_{},upper_body_joint_velocities_buffer_{};
'''
tests = r'''
};
void near(double x,double y,double tolerance=1e-10) {
  if(std::abs(x-y)>=tolerance) { std::cerr << "Mismatch: " << x << " != " << y << "\n"; }
  assert(std::abs(x-y)<tolerance);
}
void start(Harness& h) {
  h.UpdateReferenceBlend();
  h.operator_state.play=true;
  h.UpdateReferenceBlend();
}
int main() {
  for(double duration: {0.5,1.0,2.0}) {
    Harness h; h.reference_blend_duration_=duration; start(h);
    assert(h.reference_blend_active_); near(h.reference_blend_alpha_,0);
    auto raw_q = h.current_motion_->q;
    auto raw_p = h.current_motion_->p;
    const int ticks = static_cast<int>(std::round(duration/0.02));
    for(int tick=0;tick<=ticks;++tick) {
      h.current_frame_=tick;
      h.UpdateReferenceBlend();
      const double a=std::min(tick*0.02/duration,1.0);
      near(h.reference_blend_alpha_,a);
      std::vector<double> q(310),dq(310),subq(30),subdq(30),p(60),z(10);
      assert(h.GatherMotionJointPositionsMultiFrame(q,0,10,5));
      assert(h.GatherMotionJointVelocitiesMultiFrame(dq,0,10,5));
      assert(h.GatherMotionJointPositionsMultiFrame(subq,0,10,5,{0,15,30}));
      assert(h.GatherMotionJointVelocitiesMultiFrame(subdq,0,10,5,{0,15,30}));
      assert(h.GatherMotionBodyPositionsMultiFrame(p,0,10,5));
      assert(h.GatherMotionRootZPositionMultiFrame(z,0,10,5));
      for(int i=0;i<10;++i) {
        int f=std::min(tick+i*5,99);
        for(int j=0;j<31;++j) { near(q[i*31+j],j+a*0.01*f); near(dq[i*31+j],a*(0.5+j)); }
        for(int k=0;k<3;++k) {
          int j=std::array<int,3>{0,15,30}[k];
          near(subq[i*3+k],j+a*0.01*f); near(subdq[i*3+k],a*(0.5+j));
        }
        near(p[i*6],a*0.01*f); near(z[i],1+a*0.001*f);
        auto quat=h.ReadReferenceBodyQuaternions(f)[0];
        double norm=0; for(double v:quat) norm+=v*v;
        // Existing SLERP uses normalized lerp for dot > 0.9995.
        near(norm,1); near(quat[0],std::cos(a*0.005*f),2e-6); near(quat[3],std::sin(a*0.005*f),2e-6);
      }
      // Re-reading the window (or repeated T) does not advance/restart alpha.
      h.UpdateReferenceBlend(); near(h.reference_blend_alpha_,a);
      if(h.reference_blend_active_) h.reference_blend_elapsed_=std::min(duration,h.reference_blend_elapsed_+0.02);
    }
    assert(!h.reference_blend_active_);
    assert(h.current_motion_->q==raw_q && h.current_motion_->p==raw_p);
    h.current_frame_=std::min(ticks,99);
    h.operator_state.play=false; h.UpdateReferenceBlend();
    std::vector<double> dq(31); h.GatherMotionJointVelocitiesMultiFrame(dq,0,1,1);
    for(double v:dq) near(v,0);
    h.operator_state.play=true; h.UpdateReferenceBlend();
    near(h.reference_blend_alpha_,0); near(h.stand_joint_pos_[0],std::min(ticks,99)*0.01);
  }
  // Disabled and ineligible paths leave original references intact.
  for(int scenario=0;scenario<9;++scenario) {
    Harness h;
    auto motion=std::make_shared<MotionSequence>(); h.current_motion_=motion;
    if(scenario==0) h.reference_blend_duration_=0;
    if(scenario==1) { h.planner_motion_=motion; }
    if(scenario==2) motion->name="streamed";
    if(scenario==3) motion->name="temporary_motion";
    if(scenario==4) motion->mode=1;
    if(scenario==5) h.is_using_encoder_=false;
    if(scenario==6) h.has_upper_body_data_=true;
    if(scenario==7) h.has_vr_3point_data_=true;
    if(scenario==8) h.has_vr_5point_data_=true;
    start(h); assert(!h.reference_blend_active_);
    near(h.ReadReferenceJointPositions(10)[0],0.1);
    near(h.ReadReferenceJointVelocities(10)[0],0.5);
  }
  // Cancel on reset, rewind, motion switch or mode switch during playback.
  for(int scenario=0;scenario<4;++scenario) {
    Harness h; start(h); h.current_frame_=10; h.reference_blend_elapsed_=0.2;
    h.UpdateReferenceBlend(); assert(h.reference_blend_active_);
    if(scenario==0) h.reinitialize_heading_=true;
    if(scenario==1) h.current_frame_=0;
    if(scenario==2) h.current_motion_=std::make_shared<MotionSequence>();
    if(scenario==3) std::const_pointer_cast<MotionSequence>(h.current_motion_)->mode=2;
    h.UpdateReferenceBlend(); assert(!h.reference_blend_active_);
  }
  // Opposite quaternion signs must use the short path.
  Harness h; auto motion=std::make_shared<MotionSequence>();
  motion->quat[20]={-1,0,0,0}; h.current_motion_=motion; start(h);
  h.reference_blend_elapsed_=0.5; h.UpdateReferenceBlend();
  near(h.ReadReferenceBodyQuaternions(10)[0][0],1);
  std::cout << "PASS: reference windows, subsets, q/dq/body/SLERP, durations, lifecycle and bypasses\n";
}
'''
math_header = root / 'gear_sonic_deploy/src/elf3/elf3_deploy_onnx_ref/include/math_utils.hpp'
with tempfile.TemporaryDirectory(prefix='elf3_reference_blend_') as temp:
    cpp = Path(temp) / 'check.cpp'
    cpp.write_text('#include "' + str(math_header) + '"\n' + preamble + members + methods + tests)
    binary = Path(temp) / 'check'
    subprocess.run(['c++','-std=c++17','-O1','-g','-fsanitize=address,undefined',str(cpp),'-o',str(binary)],check=True)
    # LeakSanitizer cannot enumerate threads in this desktop's traced sandbox.
    # Keep address/undefined-behavior checks enabled.
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0')
    subprocess.run([str(binary)],check=True,env=env)
