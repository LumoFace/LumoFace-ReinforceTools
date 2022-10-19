#ifndef BACKPROP_TOOLS_NN_PARAMETERS_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_NN_PARAMETERS_OPERATIONS_GENERIC_H

#include "parameters.h"

namespace backprop_tools{
    template <typename DEVICE, typename CONTAINER>
    void malloc(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& p){
        malloc(device, p.parameters);
    }
    template <typename DEVICE, typename CONTAINER>
    void free(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& p){
        free(device, p.parameters);
    }
    template <typename DEVICE, typename CONTAINER>
    void malloc(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& p){
        malloc(device, (nn::parameters::Plain::instance<CONTAINER>&) p);
        malloc(device, p.gradient);
    }
    template <typename DEVICE, typename CONTAINER>
    void free(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& p){
        free(device, (nn::parameters::Plain::instance<CONTAINER>&) p);
        free(device, p.gradient);
    }
    template<typename DEVICE, typename CONTAINER>
    void zero_gradie