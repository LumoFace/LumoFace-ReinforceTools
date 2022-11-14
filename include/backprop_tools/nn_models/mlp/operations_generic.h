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
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TEMP_SPEC>
    void evaluate_memless(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<TEMP_SPEC>& layer_output_tick, Matrix<TEMP_SPEC>& layer_output_tock){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(TEMP_SPEC::ROWS >= BATCH_SIZE);
        static_assert(TEMP_SPEC::COLS >= MODEL_SPEC::HIDDEN_DIM);
        evaluate(device, network.input_layer, input, layer_output_tick);
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                evaluate(device, network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                evaluate(device, network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            evaluate(device, network.output_layer, layer_output_tick, output);
        } else {
            evaluate(device, network.output_layer, layer_output_tock, output);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_MODEL_SPEC>
    void evaluate(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC>& buffers){
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
        evaluate_memless(device, network, input, output, buffers.tick, buffers.tock);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = typename MODEL_SPEC::T;
        using TICK_TOCK_SPEC = matrix::Specification<T, typename DEVICE::index_t, BATCH_SIZE, MODEL_SPEC::HIDDEN_DIM>;
#ifndef BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
        MatrixDynamic<TICK_TOCK_SPEC> layer_output_tick;
        MatrixDynamic<TICK_TOCK_SPEC> layer_output_tock;
#else
        MatrixStatic<TICK_TOCK_SPEC> layer_output_tick;
        MatrixStatic<TICK_TOCK_SPEC> layer_output_tock;
#endif
        malloc(device, layer_output_tick);
        malloc(device, layer_output_tock);
        evaluate_memless(device, network, input, output, layer_output_tick, layer_output_tock);
        free(device, layer_output_tick);
        free(device, layer_output_tock);
    }

    // forward modifies intermediate outputs and pre activations to facilitate backward pass
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TEMP_SPEC>
    void forward_memless(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<TEMP_SPEC>& layer_output_tick, Matrix<TEMP_SPEC>& layer_output_tock){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(TEMP_SPEC::ROWS == BATCH_SIZE);
        static_assert(TEMP_SPEC::COLS == MODEL_SPEC::HIDDEN_DIM);

        forward(network.input_layer, input, layer_output_tick);
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                forward(network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                forward(network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            forward(network.output_layer, layer_output_tick, output);
        } else {
            forward(network.output_layer, layer_output_tock, output);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_MODEL_SPEC>
    void forward(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC> buffers){
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
        forward_memless(device, network, input, output, buffers.tick, buffers.tock);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC>
    void forward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input) {
        forward(device, network.input_layer, input);

        auto current_output = network.input_layer.output;
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; 