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
        free(device, buffers.next_observations);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::EpisodeStats<SPEC>& episode_stats) {
        free(device, episode_stats.data);
        episode_stats.returns._data = nullptr;
        episode_stats.steps._data = nullptr;
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        free(device, runner.buffers);
        free(device, runner.states);
        free(device, runner.episode_return);
        free(device, runner.episode_step);
        free(device, runner.truncated);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            free(device, runner.replay_buffers[env_i]);
            free(device, runner.episode_stats[env_i]);
        }
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::Batch<SPEC>& batch) {
        free(device, batch.observations_actions_next_observations);
        batch.observations.            _data = nullptr;
        batch.actions.                 _data = nullptr;
        batch.next_observations.       _data = nullptr;
        batch.observations_and_actions._data = nullptr;
        free(device, batch.rewards);
        free(device, batch.terminated);
        free(device, batch.truncated);
    }
    template<typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner, typename SPEC::ENVIRONMENT envs[SPEC::N_ENVIRONMENTS]) {
        set_all(device, runner.truncated, true);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            init(device, runner.replay_buffers[env_i]);
            runner.envs[env_i] = envs[env_i];
        }
#ifdef BACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::off_policy_runner{
        template<typename DEVICE, typename SPEC, typename RNG>
        void prologue(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
            using TI = typename DEVICE::index_t;
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
                prologue_per_env(device, &runner, rng, env_i);
            }
        }
        template<typename DEVICE, typename SPEC, typename POLICY, typename POLICY_BUFFERS>
        void interlude(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, POLICY &policy, POLICY_BUFFERS& policy_eval_buffers) {
            evaluate(device, policy, runner.buf