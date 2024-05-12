#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>
namespace bpt = backprop_tools;
#include "../../../../../src/rl/environments/mujoco/ant/ppo/parameters.h"
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
using T = float;
using TI = typename DEVICE::index_t;
using envp = parameters::environment<double, TI>;
using rlp = parameters::rl<T, TI, envp::ENVIRONMENT>;
using STATE = envp::ENVIRONMENT::State;

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_SINGLE_CORE){
    constexpr TI NUM_STEPS = 10000;

    DEVICE device;
    envp::ENVIRONMENT env;
    STATE state, next_state;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, envp::ENVIRONMENT::ACTION_DIM>> action;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    bpt::malloc(device, env);
    bpt::malloc(device, action);

    bpt::sample_initial_state(device, env, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(TI step_i = 0; step_i < NUM_STEPS; step_i++){
        bpt::step(device, env, state, action, next_state);
        if(step_i % 1000 == 0 || bpt::terminated(device, env, next_state, rng)) {
            bpt::sample_initial_state(device, env, state, rng);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Single Core Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_INDEPENDENT){
    constexpr TI NUM_STEPS_PER_THREAD = 1000;
    constexpr TI NUM_THREADS = 16;

    DEVICE device;
    envp::ENVIRONMENT envs[NUM_THREADS];
    std::thread threads[NUM_THREADS];
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, NUM_THREADS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    auto proto_rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        bpt::malloc(device, envs[env_i]);
    }
    bpt::malloc(device, actions);


    auto start = std::chrono::high_resolution_clock::now();
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i] = std::thread([&device, &rngs, &actions, &envs, env_i](){
            STATE state, next_state;
            auto rng = rngs[env_i];
            auto& env = envs[env_i];
            auto action = bpt::view(device, actions, bpt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
            bpt::randn(device, action, rng);
            bpt::sample_initial_state(device, env, state, rng);
            for(TI step_i = 0; step_i < NUM_STEPS_PER_THREAD; step_i++){
                bpt::step(device, env, state, action, next_state);
                if(step_i % 1000 == 0 || bpt::terminated(device, env, next_state, rng)) {
                    bpt::sample_initial_state(device, env, state, rng);
                }
            }
        });
    }
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_THREAD * NUM_THREADS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}


template <TI NUM_THREADS>
class TwoWayBarrier {
public:
    TwoWayBarrier() : count(0), waiting(0), waiting2(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        ++count;
        ++waiting;
        if (count < NUM_THREADS) {
            cond.wait(lock, [this] {
                return this->count == NUM_THREADS;
            });
        } else {
            cond.notify_all();
            waiting2 = 0;
        }
        --waiting;
        ++waiting2;
        if (waiting > 0) {
            cond.wait(lock, [this] {
                return this->waiting == 0;
            });
        } else {
            cond.notify_all();
            count = 0;
        }
        --waiting2;
        if (waiting2 > 0) {
            cond.wait(lock, [this] {
                return this->waiting2 == 0;
            });
        } else {
            cond.notify_all();
        }
    }

private:
    int count;
    int waiting;
    int waiting2;
    std::condition_variable cond;
    std::mutex mutex;
};
//
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_LOCKSTEP){
    constexpr TI NUM_STEPS_PER_THREAD = 1000;
    constexpr TI NUM_THREADS = 16;

    DEVICE device;
    envp::ENVIRONMENT envs[NUM_THREADS];
    std::thread threads[NUM_THREADS];
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, NUM_THREADS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    auto proto_rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        bpt::malloc(device, envs[env_i]);
        rngs[env_i] = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), env_i);
    }
    bpt::malloc(device, actions);

    TwoWayBarrier<NUM_THREADS> barrier;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> order;
    std::mutex order_mutex;
    T step_time[NUM_THREADS] = {0};
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i] = std::thread([&device, &rngs, &actions, &envs, &barrier, &order, &order_mutex, &step_time, env_i](){
            STATE state, next_state;
            auto rng = rngs[env_i];
            auto& env = envs[env_i];
            auto action = bpt::view(device, actions, bpt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
            bpt::randn(device, action, rng);
            bpt::sample_initial_state(device, env, state, rng);
            for(TI step_i = 0; step_i < NUM_STEPS_PER_THREAD; step_i++){
                bpt::randn(device, action, rng);
                bpt::step(device, env, state, action, next_state);
                if(step_i % 1000 == 0 || bpt::terminated(device, env, next_state, rng)) {
                    bpt::sample_initial_state(device, env, state, rng);
                }
                {
                    std::lock_guard<std::mutex> lock(order_mutex);
                    order.push_back(env_i);
                }
                barrier.wait();
            }
        });
    }
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_THREAD * NUM_THREADS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;


    for(TI i = 0; i < order.size()/NUM_THREADS; i++){
        bool found[NUM_THREADS] = {false};
        for(TI j = 0; j < NUM_THREADS; j++){
            found[order[i*NUM_THREADS + j]] = true;
      