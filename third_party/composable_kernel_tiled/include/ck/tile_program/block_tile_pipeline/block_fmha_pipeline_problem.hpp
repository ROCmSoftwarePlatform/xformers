// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          index_t kBlockSize_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          bool kM0NeedPadding_ /* padding for seqlen_q */,
          bool kN0K1NeedPadding_ /* padding for seqlen_k */,
          bool kSupportsBias_,
          typename BlockFmhaMask_>
struct BlockFmhaPipelineProblem
{
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using SaccDataType        = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType        = remove_cvref_t<BiasDataType_>;
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockFmhaShape      = remove_cvref_t<BlockFmhaShape_>;
    using BlockFmhaMask       = remove_cvref_t<BlockFmhaMask_>;

    static constexpr index_t kBlockSize    = kBlockSize_;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kM0NeedPadding   = kM0NeedPadding_;
    static constexpr bool kN0K1NeedPadding = kN0K1NeedPadding_;
    static constexpr bool kSupportsBias    = kSupportsBias_;
};

} // namespace block
} // namespace tile_program
} // namespace ck
