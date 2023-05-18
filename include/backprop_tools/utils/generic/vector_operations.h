
#ifndef BACKPROP_TOOLS_UTILS_GENERIC_VECTOR_OPERATIONS_H
#define BACKPROP_TOOLS_UTILS_GENERIC_VECTOR_OPERATIONS_H

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

namespace backprop_tools::utils::vector_operations{
    template <typename DEVICE, typename T, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void scalar_multiply(const T v[N], const T s, T out[N]) {
        for(typename DEVICE::index_t i = 0; i < N; i++) {
            out[i] = v[i]*s;
        }
    }

    template <typename DEVICE, typename T, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void scalar_multiply(T v[N], const T s) {
        for(typename DEVICE::index_t i = 0; i < N; i++) {
            v[i] *= s;
        }
    }

    template <typename DEVICE, typename T, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void scalar_multiply_accumulate(const T v[N], T s, T out[N]) {
        for(typename DEVICE::index_t i = 0; i < N; i++) {
            out[i] += v[i]*s;
        }
    }

    template <typename DEVICE, typename T, auto M, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void matrix_vector_product(const T A[M][N], const T v[N], T out[M]) {
        for(typename DEVICE::index_t i = 0; i < M; i++) {
            out[i] = 0;
            for(typename DEVICE::index_t j = 0; j < N; j++) {
                out[i] += A[i][j]*v[j];
            }
        }
    }

    template <typename DEVICE, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void cross_product(const T v1[3], const T v2[3], T out[3]) {
        // flops: 2 * 3 = 6
        out[0] = v1[1]*v2[2] - v1[2]*v2[1];
        out[1] = v1[2]*v2[0] - v1[0]*v2[2];
        out[2] = v1[0]*v2[1] - v1[1]*v2[0];
    }

    template <typename DEVICE, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void cross_product_accumulate(const T v1[3], const T v2[3], T out[3]) {
        out[0] += v1[1]*v2[2] - v1[2]*v2[1];
        out[1] += v1[2]*v2[0] - v1[0]*v2[2];
        out[2] += v1[0]*v2[1] - v1[1]*v2[0];
    }

    template <typename DEVICE, typename T, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void add(const T v1[N], const T v2[N], T out[N]) {
        for(typename DEVICE::index_t i = 0; i < N; i++) {
            out[i] = v1[i] + v2[i];
        }
    }
    template <typename DEVICE, typename T, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void add_accumulate(const T v1[N], const T v2[N], T out[N]) {
        for(typename DEVICE::index_t i = 0; i < N; i++) {
            out[i] += v1[i] + v2[i];
        }
    }
    template <typename DEVICE, typename T, auto N>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void add_accumulate(T const v[N], T out[N]) {
        for(typename DEVICE::index_t i = 0; i < N; i++) {
            out[i] += v[i];