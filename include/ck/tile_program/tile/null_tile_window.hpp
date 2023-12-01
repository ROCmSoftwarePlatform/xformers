// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor/tensor_view.hpp"
#include "ck/utility/common_header.hpp"

namespace ck {
namespace tile_program {

// placeholder type if we want to opt-out a tile window parameter
template <typename WindowLengths_>
struct NullTileWindow
{
    using BottomTensorView = NullTensorView;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;

    using BottomTensorIndex = Array<index_t, WindowLengths::Size()>;

    __device__ constexpr NullTileWindow() = default;

    __device__ constexpr NullTileWindow(const WindowLengths& window_lengths)
        : window_lengths_{window_lengths}
    {
    }

    __device__ constexpr auto GetWindowLengths() const { return window_lengths_; }

    __device__ constexpr auto GetBottomTensorView() const { return NullTensorView{}; }

    __device__ constexpr auto GetWindowOrigin() const { return BottomTensorIndex{}; }

    WindowLengths window_lengths_;
};

template <typename WindowLengths>
__device__ constexpr auto make_null_tile_window(const WindowLengths& window_lengths)
{
    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    return NullTileWindow<remove_cvref_t<WindowLengths>>{window_lengths};
}

template <typename WindowLengths, typename... Ts>
__device__ constexpr auto
make_tile_window(NullTensorView, const WindowLengths& window_lengths, Ts&&...)
{
    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    return NullTileWindow<remove_cvref_t<WindowLengths>>{window_lengths};
}

template <typename WindowLengths>
__device__ void move_tile_window(NullTileWindow<WindowLengths>&,
                                 const typename NullTileWindow<WindowLengths>::BottomTensorIndex&)
{
}

} // namespace tile_program
} // namespace ck
