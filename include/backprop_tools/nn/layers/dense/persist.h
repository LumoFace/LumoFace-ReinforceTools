#ifndef BACKPROP_TOOLS_NN_LAYERS_DENSE_PERSIST_H
#define BACKPROP_TOOLS_NN_LAYERS_DENSE_PERSIST_H
#include <backprop_tools/containers/persist.h>
#include "layer.h"
#include <backprop_tools/utils/persist.h>
#include <iostream>
namespace backprop_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::Layer<