
#define BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>
namespace bpt = backprop_tools;
#include "../parameters_ppo.h"
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <backprop_tools/rl/algorithms/ppo/operations_generic.h>
#include <backprop_tools/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = bpt::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
using TI = typename bpt::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: bpt::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;


using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_CUDA = bpt::DEVICE_FACTORY_GPU<bpt::devices::DefaultCUDASpecification>;
using T = float;
using TI = typename DEVICE::index_t;
using envp = parameters::environment<double, TI>;
using rlp = parameters::rl<T, TI, envp::ENVIRONMENT>;
using STATE = envp::ENVIRONMENT::State;

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_SPAWNING_CUDA){
    constexpr TI NUM_ROLLOUT_STEPS = 760;
    constexpr TI NUM_STEPS_PER_ENVIRONMENT = 64;
    constexpr TI NUM_ENVIRONMENTS = 64;
    constexpr TI NUM_THREADS = 16;
    using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, envp::ENVIRONMENT::OBSERVATION_DIM, envp::ENVIRONMENT::ACTION_DIM, 3, 256, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::IDENTITY>;
    using ACTOR_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = bpt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;

    DEVICE device;
    DEVICE_CUDA device_cuda;
    bpt::init(device_cuda);
    STATE states[NUM_ENVIRONMENTS], next_states[NUM_ENVIRONMENTS];
    envp::ENVIRONMENT envs[NUM_ENVIRONMENTS];
    ACTOR_TYPE actor_cpu, actor_gpu;
    ACTOR_TYPE::Buffers<NUM_ENVIRONMENTS> actor_buffers;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::ACTION_DIM>> actions, actions_gpu;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::OBSERVATION_DIM>> observations, observations_gpu;
    auto proto_rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    bpt::malloc(device, actions);
    bpt::malloc(device, observations);
    bpt::malloc(device, actor_cpu);
    bpt::malloc(device_cuda, actions_gpu);
    bpt::malloc(device_cuda, observations_gpu);
    bpt::malloc(device_cuda, actor_gpu);
    bpt::malloc(device_cuda, actor_buffers);
    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        bpt::malloc(device, envs[env_i]);
    }

    bpt::randn(device, actions, proto_rng);
    bpt::init_weights(device, actor_cpu, proto_rng);
    bpt::copy(device_cuda, device, actor_gpu, actor_cpu);

    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        bpt::sample_initial_state(device, envs[env_i], states[env_i], proto_rng);
        auto observation = bpt::view(device, observations, bpt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
        bpt::observe(device,envs[env_i], states[env_i], observation);
    }


    auto start = std::chrono::high_resolution_clock::now();
    for(TI rollout_step_i = 0; rollout_step_i < NUM_ROLLOUT_STEPS; rollout_step_i++){
        bpt::copy(device_cuda, device, actor_gpu, actor_cpu);
        std::cout << "Rollout step " << rollout_step_i << std::endl;
        for(TI step_i = 0; step_i < NUM_STEPS_PER_ENVIRONMENT; step_i++) {
            std::vector<std::thread> threads;
            for(TI thread_i = 0; thread_i < NUM_THREADS; thread_i++){
                threads.emplace_back([&device, &rngs, &actions, &observations, &envs, &states, &next_states, thread_i, step_i](){
                    for(TI env_i = thread_i; env_i < NUM_ENVIRONMENTS; env_i += NUM_THREADS){
                        auto rng = rngs[thread_i];
                        auto& env = envs[env_i];
                        auto& state = states[env_i];
                        auto& next_state = next_states[env_i];
                        auto action = bpt::view(device, actions, bpt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
                        auto observation = bpt::view(device, observations, bpt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
                        bpt::step(device, env, state, action, next_state);
                        if(step_i % 1000 == 0 || bpt::terminated(device, env, next_state, rng)) {
                            bpt::sample_initial_state(device, env, state, rng);
                        }
                        else{
                            next_state = state;
                        }
                        bpt::observe(device, env, next_state, observation);
                    }
                });
            }
            for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
                threads[env_i].join();
            }
            bpt::copy(device_cuda, device, observations_gpu, observations);
            bpt::evaluate(device_cuda, actor_gpu, observations_gpu, actions_gpu, actor_buffers);
            bpt::copy(device, device_cuda, actions, actions_gpu);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_ENVIRONMENT * NUM_ENVIRONMENTS * NUM_ROLLOUT_STEPS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}