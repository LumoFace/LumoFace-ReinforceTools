#ifndef BACKPROP_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
namespace backprop_tools{
    constexpr bool compile_time_redefinition_detector = true; // When importing different devices don't import the full header. The operations need to be imporeted interleaved (e.g. include cpu group 1 -> includ