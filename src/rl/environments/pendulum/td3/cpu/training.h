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

using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSp