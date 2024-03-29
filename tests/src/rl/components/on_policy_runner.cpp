#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/rl/environments/pendulum/pendulum.h>
#include <backprop_tools/rl/environments/pendulum/operations_generic.h>
#include <backprop_tools/nn_models/mlp_unconditional_stddev/operations_cpu.h>
#include <backprop_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <backprop_tools/rl/components/on_policy_runner/operations_generic.h>
#include <backprop_tools/rl/components/on_policy_runner/persist.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>


TEST(BACKPROP_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = bpt::rl::environments::pendulum::Specification<T, TI>;
    using ENVIRONMENT = bpt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;

    constexpr TI N_ENVIRONMENTS = 3;
    using ON_POLICY_RUNNER_SPEC = bpt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS>;
    using ON_POLICY_RUNNER = bpt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;


    DEVICE device;
    ON_POLICY_RUNNER runner;
    bpt::malloc(device, runner);
    ENVIRONMENT envs[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 199);
    bpt::init(device, runner, envs, rng);

    using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::TANH>;
    using ACTOR_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = bpt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
    using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;


    constexpr TI STEPS_PER_ENV = 1000;
    using DATASET_SPEC = bpt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
    using DATASET = bpt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

    ACTOR_TYPE actor;
    ACTOR_BUFFERS actor_buffers;
    DATASET dataset;
    bpt::malloc(device, actor);
    bpt::malloc(device, actor_buffers);
    bpt::malloc(device, dataset);
    bpt::init_weights(device, actor, rng);
    bpt::set_all(device, dataset.data, 0);


    bpt::collect(device, dataset, runner, actor, actor_buffers, rng);
    bpt::print(device, dataset.data);
    bpt::collect(device, dataset, runner, ac