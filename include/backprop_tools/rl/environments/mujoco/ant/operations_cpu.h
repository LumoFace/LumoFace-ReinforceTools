#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_OPERATIONS_CPU_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_OPERATIONS_CPU_H

#include "ant.h"
#include <cstring>
namespace backprop_tools::rl::environments::mujoco::ant{
    #include "model.h"
}
namespace backprop_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::mujoco::Ant<SPEC>& env) {
        using TI = typename DEVICE::index_t;
        constexpr typename DEVICE::index_t error_length = 1000;
        char error[error_length] = "Could not load model";
        {
            mjVFS* vfs = new mjVFS; // needs to be allocated on the heap because it is huge and putting it on the stack resulted in a stack overflow on windows
            mj_defaultVFS(vfs);
            mj_makeEmptyFileVFS(vfs, "model.xml", backprop_tools::rl::environments::mujoco::ant::model_xml_len);
            int file_idx = mj_findFileVFS(vfs, "model.xml");
            std::memcpy(vfs->filedata[file_idx], backprop_tools::rl::environments::mujoco::ant::model_xml, backprop_tools::rl::environments::mujoco::ant::model_xml_len);
            env.model = mj_loadXML("model.xml", vfs, error, error_length);
            mj_deleteFileVFS(vfs, "model.xml");
            delete vfs;
        }
#ifdef BACKPROP_TOOLS_DEBUG_RL_ENVIRONMENTS_MUJOCO_CHECK_INIT
        utils::assert_exit(device, env.model != nullptr, error);
#endif
        env.data = mj_makeData(env.model);
        for(TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++){
            env.init_q[state_i] = env.data->qpos[state_i];
        }
        for(TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++){
            env.init_q_dot[state_i] = env.data->qvel[state_i];
        }
        env.torso_id = mj_name2id(env.model, mjOBJ_XBODY, "torso");

    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::mujoco::Ant<SPEC>& env){
        mj_deleteData(env.data);
        mj_deleteModel(env.model);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::mujoco::Ant<SPEC>& env, typename rl::environments::mujoco::ant::State<SPEC>& state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        mj_resetData(env.model, env.data);
        for(TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++){
            state.q    [state_i] = env.init_q    [state_i] + random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -SPEC::PARAMETERS::RESET_NOISE_SCALE, SPEC::PARAMETERS::RESET_NOISE_SCALE, rng);
        }
        for(TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++){
            state.q_dot[state_i] = env.init_q_dot[state_i] + random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, SPEC::PARAMETERS::RESET_NOISE_SCALE, rng);
        }
        mj_forward(env.model, env.data);
    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const rl::environments::mujoco::Ant<SPEC>& env, typename rl::environments::mujoco::ant::State<SPEC>& state){
        using TI = typename DEVICE::index_t;
        mj_resetData(env.model, env.data);
        for(TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++){
            state.q    [state_i] = env.init_q[state_i];
        }
        for(TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++){
            state.q_dot[state_i] = env.init_q_dot[state_i];
        }
        mj_forward(env.model, env.data);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, rl::environments::mujoco::Ant<SPEC>& env, const rl::environments::mujoco::ant::State<SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::mujoco::ant::State<SPEC>& next_state) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        T x_pre = env.data->xpos[env.torso_id * 3];

