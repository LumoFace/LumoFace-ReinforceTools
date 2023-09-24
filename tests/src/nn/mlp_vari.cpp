#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/optimizers/adam/operations_generic.h>
#include <backprop_tools/nn/operations_cpu.h>
#include <backprop_tools/nn_models/operations_cpu.h>

namespace bpt = backprop_tools;

#include <gtest/gtest.h>


//template <typename T_CONTENT>
//struct OutputModule{
//    using CONTENT = T_CONTENT;
//    static constexpr auto MAX_HIDDEN_DIM = CONTENT::INPUT_DIM;
//    CONTENT content;
//};
//
//template <typename T_CONTENT, typename T_NEXT_MODULE>
//struct Specification{
//    using CONTENT = T_CONTENT;
//    using NEXT_MODULE = T_NEXT_MODULE;
//    static constexpr auto NEXT_MODULE_INPUT_DIM = NEXT_MODULE::CONTENT::INPUT_DIM;
//    static_assert(NEXT_MODULE_INPUT_DIM == CONTENT::OUTPUT_DIM);
//    static constexpr auto NEXT_MODULE_INPUT_DIM = NEXT_MODULE::CONTENT::INPUT_DIM;
//};
//

struct OutputModule{};
template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
struct Module{
    using CONTENT = T_CONTENT;
    using NEXT_MODULE = T_NEXT_MODULE;
    CONTENT content;
    NEXT_MODULE next_module;
};

namespace backprop_tools{
    template <typename DEVICE, typename CONTENT, typename NEXT_MODULE, typename INPUT, typename OUTPUT>
    void forward(DEVICE& device, Module<CONTENT, NEXT_MODULE>& module, INPUT& input, OUTPUT& output){
        forward(device, module.content, input);
        if constexpr(!bpt::utils::typing::is_same_v<NEXT_MODULE, OutputModule>){
            forward(device, module.next_module, module.content.output, output);
        }
        else{
            bpt::copy(device, device, output, module.content.output);
        }
    }
}


TEST(BACKPROP_TOOLS_NN_MODELS_MLP_VARI, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using MLP_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, 5, 2,