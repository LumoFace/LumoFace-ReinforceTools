#ifndef BACKPROP_TOOLS_NN_LOSS_FUNCTIONS_MSE_OPERATIONS_CUDA_H
#define BACKPROP_TOOLS_NN_LOSS_FUNCTIONS_MSE_OPERATIONS_CUDA_H

#include <backprop_tools/devices/cuda.h>

namespace backprop_tools::nn::loss_functions::mse {
    namespace internal::mse{
        template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
        __global__
        void d_mse_d_x_kernel(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, Matrix<SPEC_DA> d_a, typename SPEC_A::T loss_weight = 1) {
            static_assert(containers::check_structure<SPEC_A, SPEC_B>);
            static_assert(containers::check_structure<SPEC_A, SPEC_DA>);
            using T = typename SPEC_A::T;
            using TI = 