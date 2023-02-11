#ifndef BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include <backprop_tools/math/operations_generic.h>
#include "off_policy_runner.h"

#include <backprop_tools/rl/components/replay_buffer/operations_generic.h>

#include "operations_generic_per_env.h"

namespace backprop_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        malloc(device, buffers.observations);
        malloc(device, buffers.actions);
        malloc(device, buffers.next_observations);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::comp