#ifndef BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CUDA_H
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include <backprop_tools/devices/dummy.h>

namespace backprop_tools{
    namespace rl::components::off_policy_runner{
        template <typename DEVICE, typename RUNNER_SPEC, typena