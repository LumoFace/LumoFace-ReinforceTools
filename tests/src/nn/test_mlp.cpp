#include <backprop_tools/operations/cpu.h>

namespace bpt = backprop_tools;

#include <backprop_tools/nn/optimizers/adam/operations_generic.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/utils/generic/memcpy.h>
#include "../utils/utils.h"
#include <sstream>
#include <random>
#include <iostream>
#include <vector>
#include <filesystem>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include "default_network_mlp.h"
#include <backprop_tools/nn_models/persist.h>
//#define SKIP_TESTS
//#define SKIP_BACKPROP_TESTS
//#define SKIP_ADAM_TESTS
//#define SKIP_OVERFITTING_TESTS
//#define SKIP_TRAINING_TESTS


constexpr uint32_t N_WEIGHTS = ((INPUT_DIM + 1) * LAYER_1_DIM + (LAYER_1_DIM + 1) * LAYER_2_DIM + (LAYER_2_DIM + 1) * OUTPUT_DIM);


using NetworkType_1 = NetworkType;

template <typename T, typename NT>
T abs_diff_network(const NT network, const HighFive::Group g){
    T acc = 0;
    std::vector<std::vector<T>> weights;
    g.getDataSet("input_layer/weight").read(weights);
    acc += abs_diff_matrix(network.input_layer.weights.parameters, weights);
    return acc;
}

//template <typename DEVICE, typename SPEC>
//typename SPEC::T abs_diff_network(const bpt::nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC> network, const HighFive::Group g){
//    using T = typename SPEC::T;
//    T acc = 0;
//    std::vector<std::vector<T>> weights;
//    g.getDataSet("input_layer/weight").read(weights);
//    acc += abs_diff_matrix<T, LAYER_1_DIM, INPUT_DIM>(network.input_layer.weights, weights);
//    return acc;
//}
using DEVICE = NN_DEVICE;

template <typename NetworkType>
class NeuralNetworkTestLoadWeights : public NeuralNetworkTest {
protected:
    NeuralNetworkTestLoadWeights(){
        device.logger = &logger;
        model_name = "model_1";
        bpt::malloc(device, network);
        bpt::malloc(device, network_buffers);
        auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
        data_file.getDataSet("model_1/gradients/0/input_layer/weight").read(batch_0_input_layer_weights_grad);
        data_file.getDataSet("model_1/gradients/0/input_layer/bias").read(batch_0_input_layer_biases_grad);
        data_file.getDataSet("model_1/gradients/0/hidden_layer_0/weight").read(batch_0_hidden_layer_0_weights_grad);
        data_file.getDataSet("model_1/gradients/0/hidden_layer_0/bias").read(batch_0_hidden_layer_0_biases_grad);
        data_file.getDataSet("model_1/gradients/0/output_layer/weight").read(batch_0_output_layer_weights_grad);
        data_file.getDataSet("model_1/gradients/0/output_layer/bias").read(batch_0_output_layer_biases_grad);
        this->reset();
        DTYPE input[INPUT_DIM];
        DTYPE output[OUTPUT_DIM];
        standardise<DTYPE, INPUT_DIM>(X_train[0].data(), X_mean.data(), X_std.data(), input);
        standardise<DTYPE, OUTPUT_DIM>(Y_train[0].data(), Y_mean.data(), Y_std.data(), output);
        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<NN_DEVICE::index_t>>> input_matrix;
        input_matrix._data = input;
        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<NN_DEVICE::index_t>>> output_matrix;
        output_matrix._data = output;
        bpt::forward(device, network, input_matrix);
//        bpt::forward(device, network, input);
        DTYPE d_loss_d_output[OUTPUT_DIM];
        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<NN_DEVICE::index_t>>> d_loss_d_output_matrix;
        d_loss_d_output_matrix._data = d_loss_d_output;
        bpt::nn::loss_functions::mse::gradient(device, network.output_layer.output, output_matrix, d_loss_d_output_matrix);
//        bpt::nn::loss_functions::d_mse_d_x<NN_DEVICE, DTYPE, OUTPUT_DIM, 1>(device, network.output_layer.output.data, output, d_loss_d_output);
        DTYPE d_input[INPUT_DIM];
        bpt::zero_gradient(device, network);
//        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, OUTPUT_DIM>> d_loss_d_output_matrix = {d_loss_d_output};
        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<NN_DEVICE::index_t>>> d_input_matrix;
        d_input_matrix._data = d_input;
        b