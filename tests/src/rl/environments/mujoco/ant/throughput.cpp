#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>
namespace bpt = backprop_tools;
#include "../../../../../src/rl/environments/mujoco/ant/ppo/parameters.h"
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <backprop_tools/rl/algorithms/ppo/operations_generic.h>
#include <backprop_tools/rl/utils/evaluation.h>

#include <gtest/gtest.h