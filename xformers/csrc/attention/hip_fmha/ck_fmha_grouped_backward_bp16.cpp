#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_backward.h"

void grouped_backward_bp16(GroupedBackwardParams& param, hipStream_t stream) {
  if (param.custom_mask_type == 0) {
    if (param.has_attn_bias)
      grouped_backward_masktype_attnbias_dispatched<ck::bhalf_t, 0, true>::Run(
          param, stream);
    else
      grouped_backward_masktype_attnbias_dispatched<ck::bhalf_t, 0, false>::Run(
          param, stream);
  } else if (param.custom_mask_type == 1) {
    if (param.has_attn_bias)
      grouped_backward_masktype_attnbias_dispatched<ck::bhalf_t, 1, true>::Run(
          param, stream);
    else
      grouped_backward_masktype_attnbias_dispatched<ck::bhalf_t, 1, false>::Run(
          param, stream);

  } else if (param.custom_mask_type == 2) {
    if (param.has_attn_bias)
      grouped_backward_masktype_attnbias_dispatched<ck::bhalf_t, 2, true>::Run(
          param, stream);
    else
      grouped_backward_masktype_attnbias_dispatched<ck::bhalf_t, 2, true>::Run(
          param, stream);
  } else
    throw std::runtime_error("Invalid custom_mask_type value");
};
