#include <backprop_tools/devices/esp32.h>
#include <backprop_tools/nn/layers/dense/layer.h>

#include "esp_dsp.h"

namespace backprop_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(devices::esp32::DSP<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, 