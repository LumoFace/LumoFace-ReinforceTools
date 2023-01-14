#ifndef BACKPROP_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_EXTENSIONS_H
#define BACKPROP_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_EXTENSIONS_H

#include "ppo.h"
#include <backprop_tools/rl/components/on_policy_runner/on_policy_runner.h>

namespace backprop_tools{
    namespace rl::algorithms::ppo{

        template <typename PPO_SPEC>
        struct TrainingBuffersHybrid{
            using SPEC = PPO_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI BATCH_SIZE = SPEC::BATCH_SIZE;
            static constexpr TI ACTION_D