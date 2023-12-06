/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <stdexcept>

#include "ck_bool_switch.h"
#include "ck_tiled_fmha_batched_infer.h"

extern template void
run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 0>(BatchedForwardParams& param,
                                                              hipStream_t stream);

extern template void
run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 1>(BatchedForwardParams& param,
                                                              hipStream_t stream);

extern template void
run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 2>(BatchedForwardParams& param,
                                                              hipStream_t stream);

void batched_infer_fp16(BatchedForwardParams& param, hipStream_t stream)
{
    if(param.custom_mask_type == 0)
        run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 0>(param, stream);
    else if(param.custom_mask_type == 1)
        run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 1>(param, stream);
    else if(param.custom_mask_type == 2)
        run_batched_infer_masktype_attnbias_dispatched<ck::half_t, 2>(param, stream);
    else
        throw std::runtime_error("Invalid custom_mask_type value");
};
