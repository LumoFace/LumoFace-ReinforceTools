#include <backprop_tools/operations/cpu_tensorboard.h>

#include <backprop_tools/rl/environments/mujoco/ant/operations_cpu.h>
#include <backprop_tools/rl/environments/mujoco/ant/ui.h>

namespace bpt = backprop_tools;

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

namespace TEST_DEFINITIONS{
    using DEVICE = bpt::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = bpt::rl::environments::mujoco::ant::Specification<T, TI, bpt::rl: