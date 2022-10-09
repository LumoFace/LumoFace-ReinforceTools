#ifndef BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM_OPERATIONS_GENERIC_H

#include "adam.h"
#include <backprop_tools/nn/parameters/operations_generic.h>
#include <backprop_tools/utils/polyak/operations_generic.h>

namespace backprop_tools{
    template <typename DEVICE, typename CONTAINER>
    void malloc(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& p){
        malloc(device, (nn::parameters::Gradient::instance<CONTAINER>&) p);
        malloc(device, p.gradient_first_order_moment);
        malloc(device, p.gradient_second_order_moment);
    }
    template <typename DEVICE, typename CONTAINER>
    void free(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& p){
        free(device, (nn::parameters::Gradient::instance<CONTAINER>&) p);
        free(device, p.gradient_first_order_moment);
        free(device, p.gradient_second_order_moment);
    }
    template<typename DEVICE, typename CONTAINER, typename PARAMETERS>
    void update(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& parameter, nn::optimizers::Adam<PARAMETERS>& optimizer) {
        utils::polyak::update(device, parameter.gradient_first_order_moment, parameter.gradient, PARAMETERS::BETA_1);
        utils::polyak::update_squared(device, parameter.gradient_second_order_moment, parameter.gradient, PARAMETERS::BETA_2);
        gradient_descent(device, parameter, optimizer);
    }

    template<typename DEVICE, typename CONTAINER_TYPE, typename PAR