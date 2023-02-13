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
    void malloc(DEVICE& device, rl::components::off_policy_runner::EpisodeStats<SPEC>& episode_stats) {
        malloc(device, episode_stats.data);
        episode_stats.returns = view<DEVICE, typename decltype(episode_stats.data)::SPEC, SPEC::EPISODE_STATS_BUFFER_SIZE, 1>(device, episode_stats.data, 0, 0);
        episode_stats.steps   = view<DEVICE, typename decltype(episode_stats.data)::SPEC, SPEC::EPISODE_STATS_BUFFER_SIZE, 1>(device, episode_stats.data, 0, 1);
    }
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        malloc(device, runner.buffers);
        malloc(device, runner.states);
        malloc(device, runner.episode_return);
        malloc(device, runner.episode_step);
        malloc(device, runner.truncated);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            malloc(device, runner.replay_buffers[env_i]);
            malloc(device, runner.episode_stats[env_i]);
        }
    }
    template <typename DEVICE, typename BATCH_SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch) {
        using BATCH = rl::components::off_policy_runner::Batch<BATCH_SPEC>;
        using SPEC = typename BATCH_SPEC::SPEC;
        using DATA_SPEC = typename decltype(batch.observations_actions_next_observations)::SPEC;
        constexpr typename DEVICE::index_t BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        malloc(device, batch.observations_actions_next_observations);
        batch.observations             = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM                    >(device, batch.observations_actions_next_observations, 0, 0);
        batch.actions                  = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::     ACTION_DIM                    >(device, batch.observations_actions_next_observations, 0, BATCH::OBSERVATION_DIM);
        batch.next_observations        = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM                    >(device, batch.observations_actions_next_observations, 0, BATCH::OBSERVATION_DIM + BATCH::ACTION_DIM);
        batch.observations_and_actions = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM + BATCH::ACTION_DIM>(device, batch.observations_actions_next_observations, 0, 0);

        malloc(device, batch.rewards);
        malloc(device, batch.terminated);
        malloc(device, batch.truncated);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        free(device, buffers.observations);
        free(device, buffers.actions);
        free(device, buffers.n