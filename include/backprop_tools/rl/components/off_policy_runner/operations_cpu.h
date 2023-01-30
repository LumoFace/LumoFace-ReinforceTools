#ifndef BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_H
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_H

#include <thread>

#include "operations_generic_per_env.h"
namespace backprop_tools::rl::components::off_policy_runner{
    constexpr auto get_num_threads(devices::ExecutionHints hints) {
        return 1;
    }
    template<typename TI, TI NUM_THREADS>
    constexpr TI get_num_threads(rl::components::off_policy_runner::ExecutionHints<TI, NUM_THREADS