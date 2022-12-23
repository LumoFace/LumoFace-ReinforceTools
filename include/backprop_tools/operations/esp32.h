#ifndef BACKPROP_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
namespace backprop_tools{
    constexpr bool compile_time_redefinition_detector = true; // When importing different devices don't import the full header. The operations need to be imporeted interleaved (e.g. include cpu group 1 -> include cuda group 1 -> include cpu group 2 -> include cuda group 2 -> ...)
}
#endif

#include "esp32/group_1.h"
#include "esp32/group_2.h"
#include "esp32/group_3.h"