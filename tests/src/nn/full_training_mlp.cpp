
#include <backprop_tools/operations/cpu.h>
//#include <backprop_tools/operations/dummy.h>


#include <backprop_tools/nn_models/models.h>


#include <backprop_tools/nn/operations_cpu.h>
#include <backprop_tools/nn_models/operations_cpu.h>


namespace bpt = backprop_tools;
#include "../utils/utils.h"

#include <gtest/gtest.h>

#include <random>
#include <chrono>
#include <highfive/H5File.hpp>


typedef double T;


using DEVICE = bpt::devices::DefaultCPU;
//template <typename T_T>
//struct StructureSpecification{
//    typedef T_T T;
//    static constexpr typename DEVICE::index_t INPUT_DIM = 17;
//    static constexpr typename DEVICE::index_t OUTPUT_DIM = 13;
//    static constexpr int NUM_LAYERS = 3;
//    static constexpr int HIDDEN_DIM = 50;
//    static constexpr bpt::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = bpt::nn::activation_functions::GELU;
//    static constexpr bpt::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = bpt::nn::activation_functions::IDENTITY;