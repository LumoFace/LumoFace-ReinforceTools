#ifndef BACKPROP_TOOLS_NN_MODELS_MLP_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_NN_MODELS_MLP_OPERATIONS_GENERIC_H

#include <backprop_tools/nn_models/mlp/network.h>
#include <backprop_tools/nn/operations_generic.h>
#include <backprop_tools/nn/parameters/operations_generic.h>
#include <backprop_tools/nn/optimizers/adam/operations_generic.h>

namespace backprop_tools {
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
        malloc(device, network.input_layer);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            malloc(device, n