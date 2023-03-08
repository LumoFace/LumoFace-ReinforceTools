#ifndef BACKPROP_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER
#define BACKPROP_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER
namespace backprop_tools::rl::components::replay_buffer{
    template<typename T_T, typename T_TI, T_TI T_OBSERVATION_DIM, T_TI T_ACTION_DIM, T_TI T_CAPACITY, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = T_ACTION_DIM;
        static constexpr TI CAPACITY = T_CAPACITY;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

}

namespace backprop_tools::rl::components {
    template <typename T_SPEC>
    struct ReplayBuffer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;