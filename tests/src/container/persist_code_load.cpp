#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "../../../data/test_backprop_tools_container_persist_matrix.h"

constexpr bool const_declaration = false;

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> orig;
    bpt::malloc(device, orig);
    bpt::randn(device, orig, rng);
    std::cout << "orig: " << std::endl;
    bpt::print(device, orig);
    std::cout << "loaded: " << std::endl;
    bpt::print(device, matrix_1::container);

    auto abs_diff = bpt::abs_diff(device, orig, matrix_1::container);
    ASSERT_FLOAT_EQ(0, abs_diff);
}

#include "../../../data/test_backprop_tools_nn_layers_dense_persist_code.h"

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_DENSE_LAYER){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::nn::layers::dense::Layer<bpt::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, bpt::nn::activation_functions::ActivationFunction::RELU>> layer;
    bpt::malloc(device, layer);
    bpt::init_kaiming(device, layer, rng);
    bpt::increment(layer.weights.parameters, 2, 1, 10);
    auto abs_diff = bpt::abs_diff(device, layer, layer_1::layer);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_DENSE_LAYER_ADAM){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    OPTIMIZER optimizer;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::nn::layers::dense::LayerBackwardGradient<bpt::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>> layer;
    bpt::malloc(device, layer);
    bpt::init_kaiming(device, layer, rng);
    bpt::zero_gradient(device, layer);
    bpt::reset_forward_state(device, layer);
    bpt::reset_optimizer_state(device, layer, optimizer);
    bpt::randn(device, layer.weights.gradient, rng);
    bpt::randn(device, layer.weights.gradient_first_order_moment, rng);
    bpt::randn(device, layer.weights.gradient_second_order_moment, rng);
    bpt::randn(device, layer.biases.gradient, rng);
    bpt::randn(device, layer.biases.gradient_first_order_moment, rng);
    bpt::randn(device, layer.biases.gradient_second_order_moment, rng);
    bpt::increment(layer.weights.parameters, 2, 1, 10);
    bpt::increment(layer.weights.grad