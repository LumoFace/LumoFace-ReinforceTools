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
    TI smaller_than_half = 0;
    TI bigger_than_half = 0;
    long int threshold = ((long int)MAX - (long int)MIN) / 2;
    for(unsigned i = 0; i < NUM_RUNS; ++i){
        auto number = bpt::random::uniform_int_distribution(DEVICE::SPEC::RANDOM(), MIN, MAX, rng);
        if(number - MIN <= threshold) {
            smaller_than_half++;
        }
        else{
            bigger_than_half++;
        }
        ASSERT_TRUE(number >= MIN);
        ASSERT_TRUE(number <= MAX);
    }
    std::cout << "smaller_than_half: " << (float)smaller_than_half/NUM_RUNS << " bigger_than_half: " << (float)bigger_than_half/NUM_RUNS << std::endl;
}
TEST(BACKPROP_TOOLS_RANDOM_ARM, TEST_INT_UNIFORM_DISTRIBUTION) {
    test_int_uniform_distribution<0, 10>();
    test_int_uniform_distribution<-10, 10>();
    test_int_uniform_distribution<-10, 0>();
    test_int_uniform_distribution<-1000, 1000000>();
    test_int_uniform_distribution<-1, 1>();
    test_int_uniform_distribution<0, 1>();
}

template <typename T, auto MIN, auto MAX, auto DENOMINATOR, int NUM_RUNS = 10000>
void test_real_uniform_limits(){
    namespace bpt = backprop_tools;
    using DEVICE = bpt::devices::DefaultARM;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    T min = (T)MIN / (T)DENOMINATOR;
    T max = (T)MAX / (T)DENOMINATOR;
    for(unsigned i = 0; i < NUM_RUNS; ++i){
        auto number = bpt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), min, max, rng);
        if(number < min || number > max){
            std::cout << "number: " << number << " min: " << min << " max: " << max << std::endl;
        }
        ASSERT_TRUE(number >= min);
        ASSERT_TRUE(number <= max);
    }
}
TEST(BACKPROP_TOOLS_RANDOM_ARM, TEST_REAL_UNIFORM_LIMITS) {
    test_real_uniform_limits<float, 0, 10, 1000>();
    test_real_uniform_limits<float, -10, 10, 1000>();
    test_real_uniform_limits<float, -10, 0, 1000>();
    test_real_uniform_limits<float, -1000, 1000000, 1000>();
}

template <typename T, auto MIN, auto MAX, auto DENOMINATOR, int NUM_RUNS = 10000>
void test_real_uniform_distribution(){
    namespace bpt = backprop_tools;
    using DEVICE = bpt::devices::DefaultARM;
    using TI = typename DEVICE::index_t;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    T min = (T)MIN / (T)DENOMINATOR;
    T max = (T)MAX / (T)DENOMINATOR;
    TI smaller_than_half = 0;
    TI bigger_than_half = 0;

    T threshold = (max - min) / 2;
    for(unsigned i = 0; i < NUM_RUNS; ++i){
        auto number = bpt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), min, max, rng);
        if(number - min <= threshold) {
            smaller_than_half++;
        }
      