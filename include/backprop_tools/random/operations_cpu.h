#ifndef BACKPROP_TOOLS_UTILS_RANDOM_OPERATIONS_CPU_H
#define BACKPROP_TOOLS_UTILS_RANDOM_OPERATIONS_CPU_H


#include <backprop_tools/utils/generic/typing.h>

#include <random>

namespace backprop_tools::random{
    std::default_random_engine default_engine(const devices::random::CPU& dev, devices::random::CPU::index_t seed = 0){
        return std::default_random_engine(seed);
    };

    template<typename T, typename RNG>
    T uniform_int_distribution(const devices::random::CPU& dev, T low, T high, RNG& rng){
        return std::uniform_int_distribution<T>(low, high)(rng);
    }
    template<typename T, typename RNG>
    T uniform_real_distribution(const de