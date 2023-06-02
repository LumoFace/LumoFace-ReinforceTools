#ifndef BACKPROP_TOOLS_NN_UTILS_POLYAK_OPERATIONS_CUDA_H
#define BACKPROP_TOOLS_NN_UTILS_POLYAK_OPERATIONS_CUDA_H


namespace backprop_tools::utils::polyak {
    // todo: polyak factor as template parameter (reciprocal INT e.g.)
    namespace internal {
        template<typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC, bool SQUARE=false>
        __global__
        void update_kernel(Matrix<TARGET_SPEC> target, const Matrix<SOURCE_SPEC> source, const typename TARGET_SPEC::T polyak) {
            static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
            using SPEC = TARGET_SPEC;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI ROWS = SPEC::ROWS;
            constexpr TI COLS = SPEC::COLS;
            TI col_i = threadIdx.x + blockIdx.x * blockDim.x;
            TI row_i = threadIdx.y + blockIdx.y * bl