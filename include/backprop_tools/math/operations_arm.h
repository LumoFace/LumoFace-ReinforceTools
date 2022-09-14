#ifndef BACKPROP_TOOLS_MATH_OPERATIONS_ARM_H
#define BACKPROP_TOOLS_MATH_OPERATIONS_ARM_H

#include "operations_generic.h"

#include <backprop_tools/devices/arm.h>

#include <cmath>
//#include <algorithm>


namespace backprop_tools::math {

    template<typename T>
    T sqrt(const devices::math::ARM&, const T x) {
        return std::sqrt(x);
    }
    template<typename T>
    T tanh(const devices::math::ARM&, const T x) {
        return std::tanh(x);
    }
    template<typename T>
    T exp(const devices::math::ARM&, const T x) {
        return std::exp(x);
    }
    template<typename T>
    T sin(const devices::math::ARM&, const T x) {
        return std::sin(x);
    }
    template<typename T>
    T cos(const devices::math::ARM&, const T x) {
        return std::cos(x);
    }
    template<typename T>
    T acos(const devices::math::ARM&, const T x) {
        return std::acos(x);
    }
    template<typename TX, typename TY>
    auto pow(const devices::math::ARM&, const TX x, const TY y) {
        return std::pow(x, y);
    }
    template<typename T>
    auto log(const devices::math