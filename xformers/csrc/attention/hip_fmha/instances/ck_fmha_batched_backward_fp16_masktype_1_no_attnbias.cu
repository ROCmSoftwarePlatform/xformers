#include <ck.hpp>
#include "ck_fmha_batched_backward.h"

template void run_batched_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false,
    true>(BatchedBackwardParams& param, hipStream_t stream);

template void run_batched_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false,
    false>(BatchedBackwardParams& param, hipStream_t stream);
