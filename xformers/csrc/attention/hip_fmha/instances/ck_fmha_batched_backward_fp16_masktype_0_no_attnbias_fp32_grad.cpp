#include <ck/ck.hpp>
#include "ck_fmha_batched_backward.h"

template void run_batched_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false,
    true>(BatchedBackwardParams& param, hipStream_t stream);