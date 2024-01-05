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
    NeuralNetwork