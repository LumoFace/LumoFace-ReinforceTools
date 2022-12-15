// ------------ Groups 1 ------------
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
#include <backprop_tools/operations/cpu_tensorboard/group_1.h>
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_mkl/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_MKL<DEV_SPEC>;
}
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_accelerate/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_ACCELERATE<DEV_SPEC>;
}
#