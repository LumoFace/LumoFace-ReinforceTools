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
        bpt::backward(device, network, input_matrix, d_loss_d_output_matrix, d_input_matrix, network_buffers);
//        bpt::backward(device, network, input, d_loss_d_output, d_input);
    }
    void reset(){

        auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
        data_file.getDataSet(model_name + "/init/input_layer/weight").read(input_layer_weights);
        data_file.getDataSet(model_name + "/init/input_layer/bias").read(input_layer_biases);
        data_file.getDataSet(model_name + "/init/hidden_layer_0/weight").read(hidden_layer_0_weights);
        data_file.getDataSet(model_name + "/init/hidden_layer_0/bias").read(hidden_layer_0_biases);
        data_file.getDataSet(model_name + "/init/output_layer/weight").read(output_layer_weights);
        data_file.getDataSet(model_name + "/init/output_layer/bias").read(output_layer_biases);
        bpt::load(device, network.input_layer.weights.parameters, input_layer_weights);
        bpt::assign(device, network.input_layer.biases.parameters, input_layer_biases.data());
        bpt::load(device, network.hidden_layers[0].weights.parameters, hidden_layer_0_weights);
        bpt::assign(device, network.hidden_layers[0].biases.parameters, hidden_layer_0_biases.data());
        bpt::load(device, network.output_layer.weights.parameters, output_layer_weights);
        bpt::assign(device, network.output_layer.biases.parameters, output_layer_biases.data());
    }

    typename NN_DEVICE::SPEC::LOGGING logger;
    NN_DEVICE device;
    NetworkType network;
    typename NetworkType::template Buffers<> network_buffers;
    std::vector<std::vector<DTYPE>> input_layer_weights;
    std::vector<DTYPE> input_layer_biases;
    std::vector<std::vector<DTYPE>> hidden_layer_0_weights;
    std::vector<DTYPE> hidden_layer_0_biases;
    std::vector<std::vector<DTYPE>> output_layer_weights;
    std::vector<DTYPE> output_layer_biases;
    std::vector<std::vector<DTYPE>> batch_0_input_layer_weights_grad;
    std::vector<DTYPE> batch_0_input_layer_biases_grad;
    std::vector<std::vector<DTYPE>> batch_0_hidden_layer_0_weights_grad;
    std::vector<DTYPE> batch_0_hidden_layer_0_biases_grad;
    std::vector<std::vector<DTYPE>> batch_0_output_layer_weights_grad;
    std::vector<DTYPE> batch_0_output_layer_biases_grad;
};

constexpr DTYPE BACKWARD_PASS_GRADIENT_TOLERANCE (1e-8);
#ifndef SKIP_BACKPROP_TESTS
using BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS = NeuralNetworkTestLoadWeights<NetworkType_1>;
#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS, input_layer_weights) {
    DTYPE out = abs_diff_matrix(
            network.input_layer.weights.gradient,
            batch_0_input_layer_weights_grad
    );
    std::cout << "input_layer_weights diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_1_DIM * INPUT_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS, input_layer_biases) {
    DTYPE out = abs_diff_matrix(
            network.input_layer.biases.gradient,
            batch_0_input_layer_biases_grad.data()
    );
    std::cout << "input_layer_biases diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_1_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS, hidden_layer_0_weights) {
    DTYPE out = abs_diff_matrix(
            network.hidden_layers[0].weights.gradient,
            batch_0_hidden_layer_0_weights_grad
    );
    std::cout << "hidden_layer_0_weights diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_2_DIM * LAYER_1_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS, hidden_layer_0_biases) {
    DTYPE out = abs_diff_matrix(
            network.hidden_layers[0].biases.gradient,
            batch_0_hidden_layer_0_biases_grad.data()
    );
    std::cout << "hidden_layer_0_biases diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * LAYER_2_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS, output_layer_weights) {
    DTYPE out = abs_diff_matrix(
            network.output_layer.weights.gradient,
            batch_0_output_layer_weights_grad
    );
    std::cout << "output_layer_weights diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * OUTPUT_DIM * LAYER_2_DIM);
}
#endif

#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS, output_layer_biases) {
    DTYPE out = abs_diff_matrix(
            network.output_layer.biases.gradient,
            batch_0_output_layer_biases_grad.data()
    );
    std::cout << "output_layer_biases diff: " << out << std::endl;
    ASSERT_LT(out, BACKWARD_PASS_GRADIENT_TOLERANCE * OUTPUT_DIM);
}
#endif
#endif


#ifndef SKIP_ADAM_TESTS
typedef BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS BACKPROP_TOOLS_NN_MLP_ADAM_UPDATE;
#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_ADAM_UPDATE, AdamUpdate) {
    this->reset();
    bpt::nn::optimizers::Adam<bpt::nn::optimizers::adam::DefaultParametersTF<DTYPE>> optimizer;

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    std::vector<std::vector<DTYPE>> batch_0_input_layer_weights;
    std::vector<DTYPE> batch_0_input_layer_biases;
    std::vector<std::vector<DTYPE>> batch_0_hidden_layer_0_weights;
    std::vector<DTYPE> batch_0_hidden_layer_0_biases;
    std::vector<std::vector<DTYPE>> batch_0_output_layer_weights;
    std::vector<DTYPE> batch_0_output_layer_biases;
    data_file.getDataSet("model_1/weights/0/input_layer/weight").read(batch_0_input_layer_weights);
    data_file.getDataSet("model_1/weights/0/input_layer/bias").read(batch_0_input_layer_biases);
    data_file.getDataSet("model_1/weights/0/hidden_layer_0/weight").read(batch_0_hidden_layer_0_weights);
    data_file.getDataSet("model_1/weights/0/hidden_layer_0/bias").read(batch_0_hidden_layer_0_biases);
    data_file.getDataSet("model_1/weights/0/output_layer/weight").read(batch_0_output_layer_weights);
    data_file.getDataSet("model_1/weights/0/output_layer/bias").read(batch_0_output_layer_biases);
    DTYPE input[INPUT_DIM];
    DTYPE output[OUTPUT_DIM];
    standardise<DTYPE, INPUT_DIM>(&X_train[0][0], &X_mean[0], &X_std[0], input);
    standardise<DTYPE, OUTPUT_DIM>(&Y_train[0][0], &Y_mean[0], &Y_std[0], output);
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> input_matrix;
    input_matrix._data = input;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> output_matrix;
    output_matrix._data = output;
    bpt::forward(device, network, input_matrix);
    DTYPE d_loss_d_output[OUTPUT_DIM];
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> d_loss_d_output_matrix;
    d_loss_d_output_matrix._data = d_loss_d_output;
    bpt::nn::loss_functions::mse::gradient(device, network.output_layer.output, output_matrix, d_loss_d_output_matrix);
    DTYPE d_input[INPUT_DIM];
    bpt::zero_gradient(device, network);
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t>>> d_input_matrix;
    d_input_matrix._data = d_input;
    bpt::backward(device, network, input_matrix, d_loss_d_output_matrix, d_input_matrix, network_buffers);
    bpt::reset_optimizer_state(device, network, optimizer);
    bpt::update(device, network, optimizer);

    DTYPE out = abs_diff_matrix(
            network.input_layer.weights.parameters,
            batch_0_input_layer_weights
    );
    ASSERT_LT(out, 1.5e-7);
}
#endif
#endif

//#ifdef SKIP_TESTS
//TEST_F(NeuralNetworkTest, OverfitSample) {
//    this->reset();
//
//    DTYPE input[INPUT_DIM];
//    DTYPE output[OUTPUT_DIM];
//    standardise<DTYPE, INPUT_DIM>(X_train[1].data(), X_mean.data(), X_std.data(), input);
//    standardise<DTYPE, OUTPUT_DIM>(Y_train[1].data(), Y_mean.data(), Y_std.data(), output);
//    constexpr int n_iter = 1000;
//    DTYPE loss = 0;
//    reset_optimizer_state(network);
//    for (int batch_i = 0; batch_i < n_iter; batch_i++){
//        forward(network, input);
//        DTYPE d_loss_d_output[OUTPUT_DIM];
//        d_mse_d_x<DTYPE, OUTPUT_DIM>(network.output_layer.output, output, d_loss_d_output);
//        loss = mse<DTYPE, OUTPUT_DIM>(network.output_layer.output, output);
//        std::cout << "batch_i: " << batch_i << " loss: " << loss << std::endl;
//
//        zero_gradient(network);
//        DTYPE d_input[INPUT_DIM];
//        backward(network, input, d_loss_d_output, d_input);
//
//        update(network, batch_i + 1, 1);
//    }
//    ASSERT_LT(loss, 5e-10);
//
//
//}
//#endif

#ifndef SKIP_OVERFITTING_TESTS
class BACKPROP_TOOLS_NN_MLP_OVERFIT_BATCH : public BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS {
public:
    BACKPROP_TOOLS_NN_MLP_OVERFIT_BATCH() : BACKPROP_TOOLS_NN_MLP_BACKWARD_PASS(){
        model_name = "model_2";
    }
protected:

    void SetUp() override {
        NeuralNetworkTest::SetUp();
        this->reset();
    }
};
#ifndef SKIP_TESTS
TEST_F(BACKPROP_TOOLS_NN_MLP_OVERFIT_BATCH, OverfitBatch) {
    this->reset();
    bpt::nn::optimizers::Adam<bpt::nn::optimizers::adam::DefaultParametersTF<DTYPE>> optimizer;

    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::ReadOnly);
    HighFive::Group g = data_file.getGroup("model_2/overfit_small_batch");

    constexpr int n_iter = 1000;
    constexpr int batch_size = 32;
    DTYPE loss = 0;
    bpt::reset_optimizer_state(device, network, optimizer);
    {
        DTYPE diff = abs_diff_network<DTYPE>(network, data_file.getGroup(model_name+"/init"));
        std::cout << "initial diff: " << diff << std::endl;
        ASSERT_EQ(diff, 0);
    }
    for (int batch_i=0; batch_i < n_iter; batch_i++){
        uint32_t batch_i_real = 0;
        loss = 0;
        bpt::zero_gradient(device, network);
        for (int sample_i=0; sample_i < batch_size; sample_i++){
            DTYPE input[INPUT_DIM];
            DTYPE output[OUTPUT_DIM];
            standardise<DTYPE,  INPUT_DIM>(X_train[batch_i_real * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
            standardise<DTYPE, OUTPUT_DIM>(Y_train[batch_i_real * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
            bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, NN_DEVICE::index_t, 1, INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<typename NN_DEVICE::index_t>>> input_matrix;
            input_matrix._data = input;
            bpt::MatrixDynamic