
#ifndef BACKPROP_TOOLS_CONTAINERS_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_CONTAINERS_OPERATIONS_GENERIC_H

#include <backprop_tools/containers.h>
#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
    #define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

#if defined(BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS) || defined(BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC)
#include <iostream>
#include <sstream>
#endif


namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, MatrixStatic<SPEC>& matrix){
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)&matrix._data_memory[0];
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(std::is_convertible<typename SPEC::T, float>::value){
                matrix._data[i] = math::nan<typename SPEC::T>(typename DEVICE::SPEC::MATH());
            }
        }
#endif
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, MatrixStatic<SPEC>& matrix){
        // free is a no-op for statically allocated matrices
    }

#ifndef BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, MatrixDynamic<SPEC>& matrix){
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)new char[SPEC::SIZE_BYTES];
        count_malloc(device, SPEC::SIZE_BYTES);

#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(std::is_convertible<typename SPEC::T, float>::value){
                matrix._data[i] = math::nan<typename SPEC::T>(typename DEVICE::SPEC::MATH());
            }
        }
#endif
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, MatrixDynamic<SPEC>& matrix){
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
#endif
        delete matrix._data;
        matrix._data = nullptr;
    }
#endif

    template<typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI rows(const Matrix<SPEC>& m){
        return SPEC::ROWS;
    }
    template<typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI cols(const Matrix<SPEC>& m){
        return SPEC::COLS;
    }
    template<typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI row_pitch(const Matrix<SPEC>& m){
        return SPEC::ROW_PITCH;
    }
    template<typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI col_pitch(const Matrix<SPEC>& m){
        return SPEC::COL_PITCH;
    }

    template<typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT inline typename SPEC::TI index(const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        typename SPEC::TI index = row * row_pitch(m) + col * col_pitch(m);
        // bounds checking for debugging
#if defined(BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS)
        if(row >= SPEC::ROWS || col >= SPEC::COLS){
#if !defined(__CUDA_ARCH__)
            std::stringstream ss;
            ss << "index: " << row << "(" << SPEC::ROWS << "):" << col << "(" << SPEC::COLS << ") out of bounds";
            throw std::runtime_error(ss.str());
#else
            printf("index: %d(%d):%d(%d) out of bounds", row, SPEC::ROWS, col, SPEC::COLS);
#endif
        }
#endif
        return index;
    }
    template<typename SPEC>
    // todo: return reference
    BACKPROP_TOOLS_FUNCTION_PLACEMENT inline typename SPEC::T& get(const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        return m._data[index(m, row, col)];
    }
    template<typename SPEC, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT inline void set(Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col, T value){
        m._data[index(m, row, col)] = value;
    }
    template<typename SPEC, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT inline void increment(Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col, T value){
        m._data[index(m, row, col)] += value;
    }
    template<typename SPEC, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT inline void multiply(Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col, T value){
        m._data[index(m, row, col)] *= value;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void transpose(DEVICE& device, Matrix<SPEC_1>& target, Matrix<SPEC_2>& source){
        static_assert(SPEC_1::ROWS == SPEC_2::COLS);
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        for(typename SPEC_1::TI i = 0; i < SPEC_1::ROWS; i++){
            for(typename SPEC_1::TI j = 0; j < SPEC_1::COLS; j++){
                set(target, i, j, get(source, j, i));
            }
        }
    }
    namespace containers::vectorization::operators{
        template<typename DEVICE, typename T>
        inline T copy(DEVICE dev, T b){
            return b;
        }