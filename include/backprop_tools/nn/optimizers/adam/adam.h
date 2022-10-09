#ifndef BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM
#define BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM

#include <backprop_tools/nn/parameters/parameters.h>

namespace backprop_tools::nn::optimizers{
    namespace adam{
        template<typename T_T>
        struct DefaultParametersTF {
            usin