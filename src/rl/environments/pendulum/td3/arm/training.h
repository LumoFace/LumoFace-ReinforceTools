//#define BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
#define BACKPROP_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
#include <backprop_tools/operations/arm.h>

namespace bpt = backprop_tools;

#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
//#include <backprop_tools/nn/layers/dense/operations_arm/dsp.h>
#include <backprop_tools/nn/operations_generic.h>
using DEVICE = bpt::devices::arm::Generic<bpt::devices::DefaultARMSpecification>;

#include <backprop_tools/rl/environments/operations_generic.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/operations_generic.h>

#include <backprop_tools/rl/utils/evaluation.h>
#ifndef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
#include <chrono>
#include <iostream>
#endif

using