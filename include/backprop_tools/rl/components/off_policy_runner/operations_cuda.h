#ifndef BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CUDA_H
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include <backprop_tools/devices/dummy.h>

namespace backprop_tools{
    namespace rl::components::off_policy_runner{
        template <typename DEVICE, typename RUNNER_SPEC, typename BATCH_SPEC, typename RNG, bool DETERMINISTIC = false>
        __global__
        void gather_batch_kernel(const rl::components::OffPolicyRunner<RUNNER_SPEC>* runner, rl::components::off_policy_runner::Batch<BATCH_SPEC>* batch, RNG rng) {
            using BATCH = rl::components::off_policy_runner::Batch<BATCH_SPEC>;
            using T = typename RUNNER_SPEC::T;
            using TI = typename RUNNER_SPEC::TI;
            static_assert(decltype(batch->observations)::COL_PITCH == 1);
            static_assert(decltype(batch->actions)::COL_PITCH == 1);
            static_assert(decltype(batch->next_observations)::COL_PITCH == 1);
            // rng
            constexpr auto rand_max = 4294967296ULL;
            static_assert(rand_max / RUNNER_SPEC::N_ENVIRONMENTS > 100);
            static_assert(rand_max / RUNNER_SPEC::REPLAY_BUFFER_CAPACITY > 100); // so that the distribution is not skewed too much towards the remainder values

            curandState rng_state;

            TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
            curand_init(rng, batch_step_i, 0, &rng_state);

            typename DEVICE::index_t env_i = DETERMINISTIC ? 0 : curand(&rng_state) % RUNNER_SPEC::N_ENVIRONMENTS;
            auto& replay_buffer = runner->replay_buffers[env_i];
            static_assert(decltype(replay_buffer.observations)::COL_PITCH == 1);
            static_assert(decltype(replay_buffer.actions)::COL_PITCH == 1);
            static_assert(decltype(replay_buffer.next_observations)::COL_PITCH == 1);

            if(batch_step_i < BATCH_SPEC::BATCH_SIZE){
                set(batch->observations, batch_step_i, 0, get(replay_buffer.observations, batch_step_i, 0));
                typename DEVICE::index_t sample_index_max = (replay_buffer.full ? RUNNER_SPEC::REPLAY_BUFFER_CAPACITY : replay_buffer.position);
#ifdef BACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_BOUNDS
                if(sample_index_max < 1){
                    printf("sample_index_max: %d\n", sample_index_max);
                    assert(sample_index_max > 0);
                }
#endif
                typename DEVICE::index_t sample_index = DETERMINISTIC ? batch_step_i : curand(&rng_state) % sample_index_max;

                // todo: replace with smarter, coalesced copy
                for(typename DEVICE::index_t i = 0; i < BATCH::OBSERVATION_DIM; i++){
                    set(batch->observations, batch_step_i, i, get(replay_buffer.observations, sample_index, i));
                    set(batch->next_observations, batch_step_i, i, get(replay_buffer.next_observations, sample_index, i));
                }
                for(typename DEVICE::index_t i = 0; i < BATCH::ACTION_DIM; i++){
            