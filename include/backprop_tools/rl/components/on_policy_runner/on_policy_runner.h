#ifndef BACKPROP_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H
#define BACKPROP_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H

namespace backprop_tools::rl::components{
    namespace on_policy_runner{
        template <typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_N_ENVIRONMENTS = 1, T_TI T_STEP_LIMIT = 0, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
            static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        };

        template <typename T_SPEC, typename T_SPEC::TI T_STEPS_PER_ENV, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        struct DatasetSpecification{
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
            static constexpr TI STEPS_PER_ENV = T_STEPS_PER_ENV;
            static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * SPEC::N_ENVIRONMENTS;
            static constexpr TI STEPS_TOTAL_ALL = (STEPS_PER_ENV+1) * SPEC::N_ENVIRONMENTS; // +1 for the final observation
        };

        template <typename T_SPEC>
        struct Dataset{
            using SPEC = typename T_SPEC::SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI STEPS_PER_ENV = T_SPEC::STEPS_PER_ENV;
            static constexpr TI STEPS_TOTAL = T_SPEC::STEPS_TOTAL;
            // structure: OBSERVATION - ACTION - ACTION_LOG_P - REWARD - TERMINATED - TRUNCATED - VALUE - ADVANTAGE - TARGEt_VALUE
            static constexpr TI