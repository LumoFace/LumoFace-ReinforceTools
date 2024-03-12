#include <backprop_tools/operations/arm.h>
#include <gtest/gtest.h>
#include <iostream>

template <auto MIN, auto MAX, int NUM_RUNS = 10000>
void test_int_uniform_limits(){
    namespace bpt = backprop_tools;
    using DEVICE = bpt::devices::DefaultARM;
    using T = float;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    for(unsigned i = 0; i < NUM_RUNS; ++i){
        auto number = bpt::random::uniform_int_distribution(DEVICE::SPEC::RANDOM(), MIN, MAX, rng);
        ASSERT_TRUE(number >= MIN);
        ASSERT_TRUE(number <= MAX);
    }
}
TEST(BACKPROP_TOOLS_RANDOM_ARM, TEST_INT_UNIFORM_LIMITS) {
    test_int_uniform_limits<0, 10>();
    test_int_uniform_limits<-10, 10>();
    test_int_uniform_limits<-10, 0>();
    test_int_uniform_limits<-1000, 1000000>();
}

template <auto MIN, auto MAX, int NUM_RUNS = 10000>
void test_int_uniform_distribution(){
    namespace bpt = backprop_tools;
    using DEVICE = bpt::devices::DefaultARM;
    using T = float;
    using TI = typename DEVICE::index_t;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    TI smaller_than_half =