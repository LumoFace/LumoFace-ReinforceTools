
#ifndef BACKPROP_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H
#define BACKPROP_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H

#include "operations_cpu_blas.h"
#include <backprop_tools/devices/cpu_mkl.h>

namespace backprop_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void evaluate(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        evaluate((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        forward((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void backward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        backward((devices::CPU_BLAS<DEV_SPEC> &) device, layer, input, d_output, d_input);
    }
}

#endif