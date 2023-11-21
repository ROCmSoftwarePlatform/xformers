#include <ck.hpp>
#include <stdexcept>

#include "ck_bool_switch.h"
#include "ck_fmha_batched_backward.h"

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    true>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    false>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false,
    true>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false,
    false>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    true,
    true>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    true,
    false>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    true>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    false>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    true,
    true>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    true,
    false>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    false,
    true>(BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    false,
    false>(BatchedBackwardParams& param, hipStream_t stream);

void batched_backward_bp16(BatchedBackwardParams& param, hipStream_t stream) {
  BOOL_SWITCH_2(
      param.has_attn_bias,
      HAS_ATTN_BIAS,
      param.use_fp32_qkv_grad,
      USE_FP32_QKV_GRAD,
      [&] {
        if (param.custom_mask_type == 0)
          run_batched_backward_masktype_attnbias_dispatched<
              ck::bhalf_t,
              0,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>(param, stream);
        else if (param.custom_mask_type == 1)
          run_batched_backward_masktype_attnbias_dispatched<
              ck::bhalf_t,
              1,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>(param, stream);
        else if (param.custom_mask_type == 2)
          run_batched_backward_masktype_attnbias_dispatched<
              ck::bhalf_t,
              2,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>(param, stream);
        else
          throw std::runtime_error("Invalid custom_mask_type value");
      });
};
