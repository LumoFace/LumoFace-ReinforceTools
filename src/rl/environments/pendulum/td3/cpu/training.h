#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
namespace bpt = backprop_tools;




#include <backprop_tools/rl/environments/operations_generic.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/operations_generic.h>


#include <backprop_tools/rl/utils/evaluation.h>

#include <filesystem>


#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
#include <backprop_tools/rl/environments/pendulum/ui.h>
#include <backprop_tools/rl/utils/evaluation_visual.h>
#endif


#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif

#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
    using LOGGER = bpt::devices::logging::CPU_TENSORBOARD;
#else
    using LOGGER = bpt::devices::logging::CPU;
#endif

#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) || defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) || defined(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
#else
using DEVICE = bpt::devices::DefaultCPU;
#endif


using T = float;
using TI = typename DEVICE::index_t;

typedef bpt::rl::environments::pendulum::Specification<T, TI, bpt::rl::environments::pendulum::DefaultParameters<T>> PENDULUM_SPEC;
typedef bpt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef bpt::rl::environments::pendulum::UI<T> UI;
#endif


struct TD3_PENDULUM_PARAMETERS: bpt::rl::algorithms::td3::DefaultParameters<T, TI>{
    constexpr static TI CRITIC_BATCH_SIZE = 100;
    constexpr static TI ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3_PENDULUM_PARAMETERS;

using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG
constexpr TI STEP_LIMIT = 1000;
#else
constexpr TI STEP_LIMIT = 10000;
#endif

constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
constexpr TI EPISODE_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<
        T,
        TI,
        ENVIRONMENT,
        1,
        REPLAY_BUFFER_CAP,
        EPISODE_STEP_LIMIT,
        bpt::rl::components::off_policy_runner::DefaultParameters<T>
>;
using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
OFF_POLICY_RUNNER_TYPE off_policy_runner;
ACTOR_CRITIC_TYPE actor_critic;
const T STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);