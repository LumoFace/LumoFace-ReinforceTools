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
            static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
            static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> actions;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> observations;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> d_action_log_prob_d_action;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> d_observations;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> target_values;
        };
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::ppo::TrainingBuffersHybrid<SPEC>& buffers){
        malloc(device, buffers.actions);
        malloc(device, buffers.observations);
        malloc(device, buffers.d_action_log_prob_d_action);
        malloc(device, buffers.d_observations);
        malloc(device, buffers.target_values);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::ppo::TrainingBuffersHybrid<SPEC>& buffers){
        free(device, buffers.actions);
        free(device, buffers.observations);
        free(device, buffers.d_action_log_prob_d_action);
        free(device, buffers.d_observations);
        free(device, buffers.target_values);
    }
    template <typename DEVICE, typename DEVICE_EVALUATION, typename PPO_SPEC, typename OPR_SPEC, auto STEPS_PER_ENV, typename ACTOR_OPTIMIZER, typename CRITIC_OPTIMIZER, typename RNG>
    void train_hybrid(DEVICE& device,
        DEVICE_EVALUATION& device_evaluation,
        rl::algorithms::PPO<PPO_SPEC>& ppo,
        rl::algorithms::PPO<PPO_SPEC>& ppo_evaluation,
        rl::components::on_policy_runner::Dataset<rl::components::on_policy_runner::DatasetSpecification<OPR_SPEC, STEPS_PER_ENV>>& dataset,
        ACTOR_OPTIMIZER& actor_optimizer,
        CRITIC_OPTIMIZER& critic_optimizer,
        rl::algorithms::ppo::Buffers<PPO_SPEC>& ppo_buffers,
        rl::algorithms::ppo::TrainingBuffersHybrid<PPO_SPEC>& hybrid_buffers,
        typename PPO_SPEC::ACTOR_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& actor_buffers,
        typename PPO_SPEC::CRITIC_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& critic_buffers,
        RNG& rng){
#ifdef BACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        utils::assert_exit(device, ppo.initialized, "PPO not initialized");
#endif
        using T = typename PPO_SPEC::T;
 