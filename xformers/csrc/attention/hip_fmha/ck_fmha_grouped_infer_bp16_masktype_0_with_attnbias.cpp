#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_infer.h"

template struct grouped_infer_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true>;
