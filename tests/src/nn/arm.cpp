#include <backprop_tools/operations/cpu/group_1.h>
#include <backprop_tools/operations/arm/group_1.h>
#include <backprop_tools/operations/cpu/group_2.h>
#include <backprop_tools/operations/arm/group_2.h>
#include <backprop_tools/operations/cpu/group_3.h>
#include <backprop_tools/operations/arm/group_3.h>

#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/operations_generic.h>

namespace bpt = backprop_tools;

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
//#include "../../../data/test_backprop_tools_nn_models_mlp_persist_code.h"

constexpr bool const_declaration = false;


template <typename DTYPE, auto INPUT_DIM, auto OUTPUT_DIM, auto N_HIDDEN_LAYERS, auto HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION, bpt::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION, auto BATCH_SIZE>
void test_mlp_evaluate() {
    using DEVICE = bpt::devices::DefaultCPU;
    using DEVICE_ARM = bpt::devices::DefaultARM;
    DEVICE device;
    DEVICE_ARM device_arm;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    using STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, INPUT_DIM, OUTPUT_DIM, N_HIDDEN_LAYERS, HIDDEN_DIM, HIDDEN_ACTIVATION_FUNCTION, ACTIVATION_FUNCTION, 1, bpt::MatrixDynamicTag, true, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
    using SPEC = bpt::nn_models::mlp::InferenceSpecification<STRUCTURE_SPEC>;
    bpt::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    bpt::malloc(device, mlp);
    bpt::init_weights(device, mlp, rng);

    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::INPUT_DIM>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::OUTPUT_DIM>> output_orig, output_arm;
    bpt::malloc(device, input);
    bpt::malloc(device, output_orig);
    bpt::malloc(device, output_arm);
    bpt::randn(device, input, rng);
    bpt::evaluate(device, mlp, input, output_orig);
    bpt::evaluate(device_arm, mlp, input, output_arm);
    bpt::print(device, output_orig);

    auto abs_diff = bpt::abs_diff(device, output_or