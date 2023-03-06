#ifndef BACKPROP_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H

#include "replay_buffer.h"
#include <backprop_tools/utils/generic/memcpy.h>

namespace backprop_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb) {
        using DATA_SPEC = typename decltype(rb.data)::SPEC;
        malloc(device, rb.data);
        rb.observations      = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, SPEC::OBSERVATION_DIM>(device, rb.data, 0, 0);
        rb.actions           = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, SPEC::ACTION_DIM     >(device, rb.data, 0, SPEC::OBSERVATION_DIM);
        rb.rewards           = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, 1                    >(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM);
        rb.next_observations = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, SPEC::OBSERVATION_DIM>(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1);
        rb.terminated        = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, 1                    >(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1 + SPEC::OBSERVATION_DIM);
        rb.truncated         = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, 1                    >(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1 + SPEC::OBSERVATION_DIM + 1);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb) {
        free(device, rb.data);
    }
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb) {
        rb.full = false;
        rb.position =