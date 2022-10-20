#ifndef BACKPROP_TOOLS_NN_PARAMETERS_PERSIST_H
#define BACKPROP_TOOLS_NN_PARAMETERS_PERSIST_H

#include <backprop_tools/nn/parameters/parameters.h>

#include <highfive/H5Group.hpp>
namespace backprop_tools{
    template<typename DEVICE, typename CONTAINER>
    v