#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "../../../data/test_backprop_tools_container_persist_matrix.h"

constexpr bool const_declaration = false