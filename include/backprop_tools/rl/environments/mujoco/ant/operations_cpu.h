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
        utils::assert_exit(device, env.model != nullptr,