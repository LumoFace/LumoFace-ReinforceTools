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
            malloc(device, network.hidden_layers[layer_i]);
        }
        malloc(device, network.output_layer);
    }
    template<typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>& buffers) {
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
    }
    template<typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetworkBuffersForwardBackward<BUFFER_SPEC>& buffers) {
        malloc(device, (nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>&) buffers);
        malloc(device, buffers.d_input);
        malloc(device, buffers.d_output);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
        free(device, network.input_layer);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            free(device, network.hidden_layers[layer_i]);
        }
        free(device, network.output_layer);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetworkBuffers<SPEC>& buffers) {
        free(device, buffers.tick);
        free(device, buffers.tock);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetworkBuffersForwardBackward<SPEC>& buffers) {
        free(device, (nn_models::mlp::NeuralNetworkBuffers<SPEC>&) buffers);
        free(device, buffers.d_input);
        free(device, buffers.d_output);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, RNG& rng) {
        init_kaiming(device, network.input_layer, rng);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            init_kaiming(device, network.hidden_layers[layer_i], rng);
        }
        init_kaiming(device, network.output_layer, rng);
    }

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers

    namespace nn_models::mlp{
        template <typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        constexpr bool check_input_output =
                INPUT_SPEC::COLS == MODEL_SPEC::INPUT_DIM &&
                INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS &&
                OUTPUT_SPEC::COLS == MODEL_SPEC::OUTPUT_DIM &&
                (!MODEL_SPEC::ENFORCE_FLOATING_POINT_TYPE || ( utils::typing::is_same_v<typename MODEL_SPEC::T, typename INPUT_SPEC::T> && utils::typing::is_same_v<typename INPUT_SPEC::T, typename OUTPUT_SPEC::T>));
   