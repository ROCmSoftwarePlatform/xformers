#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_infer.h"

template struct batched_infer_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true>;

template struct batched_infer_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false>;
