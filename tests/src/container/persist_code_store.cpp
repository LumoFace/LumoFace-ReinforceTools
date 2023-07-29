#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/optimizers/adam/persist_code.h>
#include <backprop_tools/nn/parameters/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn/layers/dense/persist_code.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/persist_code.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>


constexpr bool const_declaration = true;


TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> m;
    bpt::malloc(device, m);
    bpt::randn(device, m, rng);
    bpt::print(device, m);
    auto output 