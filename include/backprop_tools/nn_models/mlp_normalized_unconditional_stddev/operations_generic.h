#include "network.h"
#include <backprop_tools/nn_models/mlp/operations_generic.h>


namespace backprop_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m){
        malloc(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m);
        malloc(device, m.log_std);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m){
        free(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m);
        free(device, m.log_std);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m, RNG& rng){
        init_weights(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m, rng);
        set_all(device, m.log_std.parameters, 0);
    }
    template<typename DEVICE, typename SPEC, typename ADAM_PARAMETERS>
    void update(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& network, nn::optimizers::Adam<ADAM_PARAMETERS>& optimizer) {
        using T = typename SPEC::T;
        optimizer.first_order_moment_bias_correction  = 1/(1 - math::pow(typename DEVICE::SPEC::MATH(), ADAM_PARAMETERS::BETA_1, (T)network.age));
        optimizer.second_order_moment_bias_correction = 1/(1 - math::pow(typename DEVICE::SPEC::MATH(), ADAM_PARAMETERS::BETA_2, (T)network.age));
        update(device, network.log_std, optimizer);
        update(device, (nn_mode