#include <backprop_tools/nn/nn.h>
#include "../utils/utils.h"


namespace bpt = backprop_tools;

using DTYPE = double;


using NN_DEVICE = bpt::devices::DefaultCPU;
using StructureSpecification = bpt::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::in