
// Group 1
#include <backprop_tools/operations/cpu/group_1.h>
//#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_CUDA
//    #include <backprop_tools/operations/cuda/group_1.h>
//#endif
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
    #include <backprop_tools/operations/cpu_mkl/group_1.h>
#endif

// Group 2
#include <backprop_tools/operations/cpu/group_2.h>
//#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_CUDA
//#include <backprop_tools/operations/cuda/group_2.h>
//#endif
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_2.h>
#endif

// Group 3
#include <backprop_tools/operations/cpu/group_3.h>
//#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_CUDA
//#include <backprop_tools/operations/cuda/group_3.h>
//#endif
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_3.h>
#endif

#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/nn_models/operations_cpu.h>

namespace bpt = backprop_tools;

#include <gtest/gtest.h>

#include <chrono>



using DTYPE = float;
using DEVICE = bpt::devices::DefaultCPU;
using INDEX_TYPE = DEVICE::index_t;

constexpr DEVICE::index_t BATCH_SIZE = 256;
constexpr DEVICE::index_t HIDDEN_DIM = BATCH_SIZE;

template <typename T, typename TI, bpt::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using StructureSpecification = bpt::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, bpt::nn::activation_functions::RELU, BATCH_SIZE>;

using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
template <typename T, typename TI, bpt::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using InferenceSpecification = bpt::nn_models::mlp::AdamSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>>;

using NetworkType = bpt::nn_models::mlp::NeuralNetworkAdam<InferenceSpecification<DTYPE, DEVICE::index_t, bpt::nn::activation_functions::RELU>>;


constexpr INDEX_TYPE ITERATIONS = 1;
constexpr INDEX_TYPE NAIVE_ITERATIONS = 1;

class BACKPROP_TOOLS_NN_DENSE_BENCHMARK : public ::testing::Test
{
protected:
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> expected_output_input_layer;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::OUTPUT_DIM>> expected_output_output_layer;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::OUTPUT_DIM>> output_target;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;

    OPTIMIZER optimizer;


    NetworkType network;
    NetworkType::BuffersForwardBackward<BATCH_SIZE> network_buffers;
    NetworkType network_mkl;
    NetworkType::BuffersForwardBackward<BATCH_SIZE> network_mkl_buffers;
    BACKPROP_TOOLS_NN_DENSE_BENCHMARK(){
        device.logger = &logger;
        bpt::malloc(device, input);
        bpt::malloc(device, expected_output_input_layer);
        bpt::malloc(device, expected_output_output_layer);
        bpt::malloc(device, output_target);
        auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
        bpt::malloc(device, network);
        bpt::malloc(device, network_buffers);
        network.age = 100000;
        bpt::malloc(device, network_mkl);
        bpt::malloc(device, network_mkl_buffers);
        bpt::init_weights(device, network, rng);
        bpt::copy(device, device, network_mkl, network);
        assert(network_mkl.age == network.age);

        for(INDEX_TYPE i = 0; i < BATCH_SIZE; ++i){
            for(INDEX_TYPE j = 0; j < NetworkType::INPUT_DIM; ++j){
                set(input, i, j, bpt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng));
            }
        }
        for(INDEX_TYPE i = 0; i < BATCH_SIZE; ++i){
            for(INDEX_TYPE j = 0; j < NetworkType::OUTPUT_DIM; ++j){
                set(output_target, i, j, bpt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng));
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for(INDEX_TYPE iteration_i = 0; iteration_i < NAIVE_ITERATIONS; iteration_i++) {
            bpt::evaluate(device, network.input_layer, input, expected_output_input_layer);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "LIC: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

        bpt::forward(device, network, input);

        bpt::copy(device, device, expected_output_output_layer, network.output_layer.output);

        bpt::reset_optimizer_state(device, network, optimizer);
        bpt::zero_gradient(device, network);
        {
            auto start = std::chrono::high_resolution_clock::now();
            for(INDEX_TYPE iteration_i = 0; iteration_i < NAIVE_ITERATIONS; iteration_i++) {
                bpt::forward_backward_mse(device, network, input, output_target, network_buffers);
            }
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "LIC forward backward mse: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;
        }
        bpt::zero_gradient(device, network);
        bpt::forward_backward_mse(device, network, input, output_target, network_buffers);
    }
};





TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, BENCHMARK_BATCH) {
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_batch;
    bpt::malloc(device, output_batch);
    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < NAIVE_ITERATIONS; iteration_i++) {
        bpt::evaluate(device, network.input_layer, input, output_batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "LIC: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;


    DTYPE abs_diff = bpt::abs_diff(device, output_batch, expected_output_input_layer);

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}

#define min(x,y) (((x) < (y)) ? (x) : (y))

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/devices/cpu_mkl.h>
#include <backprop_tools/containers/operations_cpu_mkl.h>
TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, MKL) {
    using T = DTYPE;
    int m, n, k;
    T alpha, beta;

    m = BATCH_SIZE;
    k = NetworkType::INPUT_DIM;
    n = NetworkType::SPEC::HIDDEN_DIM;
    // A m x k
    // B k x n
    // C m x n
    alpha = 1.0; beta = 0.0;

    constexpr INDEX_TYPE alignment = 64;

    bpt::MatrixDynamic<bpt::matrix::Specification<T, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_mkl_matrix;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, DEVICE::index_t, BATCH_SIZE, NetworkType::SPEC::HIDDEN_DIM>> output_mkl_matrix;
    bpt::malloc(device, input_mkl_matrix);
    bpt::malloc(device, output_mkl_matrix);
    bpt::copy(device, device, input_mkl_matrix, input);
    bpt::set_all(device, output_mkl_matrix, (T)0);

    static_assert(decltype(network_mkl.input_layer.weights.parameters)::COL_PITCH == 1);
    static_assert(decltype(input_mkl_matrix)::COL_PITCH == 1);
    static_assert(decltype(output_mkl_matrix)::COL_PITCH == 1);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        if constexpr(bpt::utils::typing::is_same_v<T, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (float*)input_mkl_matrix._data, bpt::row_pitch(input_mkl_matrix), (float*)network_mkl.input_layer.weights.parameters._data, bpt::row_pitch(network_mkl.input_layer.weights.parameters), beta, (float*)output_mkl_matrix._data, bpt::row_pitch(output_mkl_matrix));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (double*)input_mkl_matrix._data, bpt::row_pitch(input_mkl_matrix), (double*)network_mkl.input_layer.weights.parameters._data, bpt::row_pitch(network_mkl.input_layer.weights.parameters), beta, (double*)output_mkl_matrix._data, bpt::row_pitch(output_mkl_matrix));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;


    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE output_i=0; output_i < HIDDEN_DIM; output_i++){
            set(output_mkl_matrix, batch_i, output_i, bpt::activation<DEVICE::SPEC::MATH, T, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>(get(output_mkl_matrix, batch_i, output_i) + get(network_mkl.input_layer.biases.parameters, 0, output_i)));
        }
    }

    T abs_diff = bpt::abs_diff(device, output_mkl_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}

#include <backprop_tools/nn/operations_cpu_mkl.h>
#include <backprop_tools/containers/operations_generic.h>
#include <backprop_tools/utils/generic/typing.h>

TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, MKL_LAYER) {
    using DEVICE_MKL = bpt::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl;
    device_mkl.logger = &logger;

    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_matrix;
    bpt::malloc(device_mkl, output_matrix);
    bpt::set_all(device_mkl, output_matrix, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        bpt::evaluate(device_mkl, network_mkl.input_layer, input, output_matrix);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC evaluate: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = bpt::abs_diff(device_mkl, output_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(bpt::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}

TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, MKL_LAYER_FORWARD) {
    std::cout << "Layer batch size: " << decltype(network_mkl.input_layer)::SPEC::BATCH_SIZE << std::endl;
    using DEVICE_MKL = bpt::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl;
    device_mkl.logger = device.logger;

    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_matrix;
    bpt::malloc(device_mkl, output_matrix);
    bpt::set_all(device_mkl, output_matrix, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        bpt::forward(device_mkl, network_mkl.input_layer, input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC forward: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = bpt::abs_diff(device_mkl, network_mkl.input_layer.output, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(bpt::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}

TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, MKL_MODEL_FORWARD) {
    std::cout << "Layer batch size: " << decltype(network_mkl.input_layer)::SPEC::BATCH_SIZE << std::endl;
    using DEVICE_MKL = bpt::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl;
    device_mkl.logger = device.logger;

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        bpt::forward(device_mkl, network_mkl, input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC forward full: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = bpt::abs_diff(device_mkl, network_mkl.output_layer.output, expected_output_output_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(bpt::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}

TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, MKL_MODEL_BACKWARD) {
    using DEVICE_MKL = bpt::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl;
    device_mkl.logger = device.logger;
    OPTIMIZER optimizer;

    bpt::reset_optimizer_state(device_mkl, network_mkl, optimizer);
    bpt::zero_gradient(device_mkl, network_mkl);
    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        bpt::forward_backward_mse(device_mkl, network_mkl, input, output_target, network_mkl_buffers);
    }
    auto end = std::chrono::high_resolution_clock::now();
    bpt::zero_gradient(device_mkl, network_mkl);
    bpt::forward_backward_mse(device_mkl, network_mkl, input, output_target, network_mkl_buffers);
    std::cout << "MKL LIC forward backward mse: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

//    DTYPE abs_diff = bpt::abs_diff(device_mkl, network_mkl.output_layer.d_weights, network.output_layer.d_weights) / NetworkType::NUM_WEIGHTS;

    DTYPE abs_diff = bpt::abs_diff(device_mkl, network_mkl, network) / NetworkType::NUM_WEIGHTS;

    if constexpr(bpt::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-4);
    }
    else{
        EXPECT_LT(abs_diff, 1e-12);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}
#endif

//#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
//#include <Accelerate/Accelerate.h>
//TEST_F(BACKPROP_TOOLS_NN_DENSE_BENCHMARK, ACCELERATE) {
//    DTYPE *A, *B, *C;
//    int m, n, k;
//    DTYPE alpha, beta;
//
//    m = HIDDEN_DIM, k = HIDDEN_DIM, n = BATCH_SIZE;
//    // A m x k
//    // B k x n
//    // C m x n
//    alpha = 1.0; beta = 0.0;
//
//    using INPUT_LAYER_WEIGHTS_SPEC = decltype(network_mkl.input_layer.weights)::SPEC;
//
//    A = (DTYPE *)malloc(INPUT_LAYER_WEIGHTS_SPEC::SIZE_BYTES);
//    B = (DTYPE *)malloc( k*n*sizeof( DTYPE ));
//    C = (DTYPE *)malloc( m*n*sizeof( DTYPE ));
//
//    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
//        for(INDEX_TYPE output_i = 0; output_i < HIDDEN_DIM; output_i++){
//            C[output_i * BATCH_SIZE + batch_i] = 0;
//        }
//    }
//
//    memcpy(A, network_mkl.input_layer.weights._data, INPUT_LAYER_WEIGHTS_SPEC::SIZE_BYTES);
//
//    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_mkl_matrix({B});
//
//    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_lic_matrix;
//    bpt::malloc(device, input_lic_matrix);
//    bpt::copy(device, device, input_lic_matrix, input);
//    DTYPE input_abs_diff = bpt::abs_diff(device, input_mkl_matrix, input_lic_matrix);
//
//    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, NetworkType::INPUT_DIM, BATCH_SIZE>> input_lic_matrix_transpose;
//    bpt::malloc(device, input_lic_matrix_transpose);
//    bpt::transpose(device, input_lic_matrix_transpose, input_lic_matrix);
//
//    memcpy(B, input_lic_matrix_transpose.data, k*n*sizeof( DTYPE ));
//
//    auto start = std::chrono::high_resolution_clock::now();
//    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
//        if constexpr(bpt::utils::typing::is_same_v<DTYPE, float>){
//            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (float*)A, k, (float*)B, n, beta, (float*)C, n);
//        }
//        else{
//            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (double*)A, k, (double*)B, n, beta, (double*)C, n);
//        }
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    std::cout << "ACCELERATE: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;
//
//
//    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
//        for(INDEX_TYPE output_i=0; output_i < HIDDEN_DIM; output_i++){
//            C[batch_i + output_i * BATCH_SIZE] = bpt::activation<DEVICE::SPEC::MATH, DTYPE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>(C[batch_i + output_i * BATCH_SIZE] + network_mkl.input_layer.biases.data[output_i]);
//        }
//    }
//
//    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM, BATCH_SIZE>> output_mkl_matrix_transpose;
//    output_mkl_matrix_transpose.data = C;
//
//    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_mkl_matrix;
//    bpt::malloc(device, output_mkl_matrix);
//
//    bpt::transpose(device, output_mkl_matrix, output_mkl_matrix_transpose);
//
//    DTYPE abs_diff = bpt::abs_diff(device, output_mkl_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;
//
//    std::cout << "Absolute difference: " << abs_diff << std::endl;
//    EXPECT_LT(abs_diff, 1e-6);
//
//}
//#endif