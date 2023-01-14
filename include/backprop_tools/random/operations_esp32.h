#ifndef BACKPROP_TOOLS_UTILS_RANDOM_OPERATIONS_ESP32_H
#define BACKPROP_TOOLS_UTILS_RANDOM_OPERATIONS_ESP32_H


#include <backprop_tools/utils/generic/typing.h>

namespace backprop_tools::random{
   devices::random::ESP32::index_t default_engine(const devices::random::ESP32& dev, devices::random::ESP32::index_t seed = 1){
       return 0b10101010101010101010101010101010 + seed;
   };
   template<typename RNG>
   void next(const devices::random::ESP32& dev, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ESP3