
#include <backprop_tools/containers/persist_code.h>
#include "parameters.h"
#include <sstream>
namespace backprop_tools {
    std::string get_type_string(nn::parameters::Plain p){
        return "backprop_tools::nn::parameters::Plain";
    }
    std::string get_type_string(nn::parameters::Gradient p){
        return "backprop_tools::nn::parameters::Gradient";
    }
    template<typename DEVICE, typename CONTAINER>
    persist::Code save_split(DEVICE &device, nn::parameters::Plain::instance<CONTAINER>& parameter, std::string name, bool const_declaration=false, typename DEVICE::index_t indent=0, bool output_memory_only=false){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        ss << ind << "namespace " << name << " {\n";
        auto container = save_split(device, parameter.parameters, "parameters_memory", const_declaration, indent+1);
        ss_header << container.header;
        ss << container.body;
        if(!output_memory_only){
            ss << ind << "    " << (const_declaration ? "const " : "") << "backprop_tools::nn::parameters::Plain::instance<parameters_memory::CONTAINER_TYPE> parameters = {parameters_memory::container};\n";
        }
        ss << ind << "}\n";
        return {"", ss.str()};
    }

    template<typename DEVICE, typename CONTAINER>
    persist::Code save_split(DEVICE &device, nn::parameters::Gradient::instance<CONTAINER>& parameter, std::string name, bool const_declaration=false, typenam