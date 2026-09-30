# ELF3 deployment package notes

This package is the corrected `gear_sonic_deploy/src/elf3` tree for the current 31-DoF ELF3 configuration.

## Corrected

- 31-DoF actuator Kp/Kd, effort limits and action scales synchronized with `gear_sonic/envs/manager_env/robots/elf3.py`.
- Active IsaacLab <-> MuJoCo DoF mappings retained (they match the current `elf3.py`).
- Wrist IsaacLab indices retained as `[15, 16, 19, 20, 23, 24]`.
- Full-joint observation registry dimensions changed from hard-coded G1 29-DoF values to expressions based on `ELF3_NUM_MOTOR`.
- Two G1 29-joint temperature-name tables replaced by 31 ELF3 names.
- `freq_test` motor count corrected to 31.
- FK consistency check corrected.
- FK unit-test MJCF path corrected to `elf3/mjcf/elf3.xml`.
- ASAN CMake flags moved after target creation.

## Intentionally retained

The external upper-body streaming API remains **17 DoF** (waist 3 + left arm 7 + right arm 7), excluding the two head joints. The main policy/action path is still 31 DoF. Do not change these payloads to 19 DoF unless the ZMQ/ROS/VLA producer and consumer are changed together.

The low-level DDS message/topics remain Unitree-G1-compatible for the current MuJoCo sim2sim bridge. They are not a final ELF3 hardware SDK implementation.

## Integration

Ensure the parent `gear_sonic_deploy/src/CMakeLists.txt` contains:

```cmake
add_subdirectory(elf3)
```

Then rebuild from `gear_sonic_deploy` before starting deployment.
