#include <backprop_tools/rl/environments/mujoco/ant/operations_cpu.h>
#include <backprop_tools/rl/algorithms/ppo/ppo.h>
#include <backprop_tools/rl/components/on_policy_runner/on_policy_runner.h>
namespace parameters_0{
    template <typename T, typename TI>
    struct environment{
        using ENVIRONMENT_PARAMETERS = bpt::rl::environments::mujoco::ant::DefaultParameters<T, TI>;
        using ENVIRONMENT_SPEC = bpt::rl::environments::mujoco::ant::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
        using ENVIRONME