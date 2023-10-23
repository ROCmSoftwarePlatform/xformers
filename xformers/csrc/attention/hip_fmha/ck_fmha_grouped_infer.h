#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_infer_xdl_cshuffle.hpp"

#include "ck_fmha_device_gemm_constants.h"
#include "ck_fmha_op_helper.h"
#include "ck_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
struct grouped_infer_masktype_attnbias_dispatched {
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;

  using GemmDataType = scalar_t;
  using ADataType = scalar_t;
  using B0DataType = scalar_t;
  using B1DataType = scalar_t;
  using AccDataType = F32;
  using CShuffleDataType = F32;
  using CDataType = scalar_t;
  using ZDataType = unsigned short;
  using LSEDataType = F32;
  using Acc0BiasDataType =
      typename std::conditional<has_attn_bias, scalar_t, void>::type;
  using Acc1BiasDataType = void;

  using AElementOp = PassThrough;
  using B0ElementOp = PassThrough;
  using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
  using B1ElementOp = PassThrough;
  using CElementOp = PassThrough;

  static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
  static constexpr auto MaskingSpec =
      static_cast<ck::tensor_operation::device::MaskingSpecialization>(
          custom_mask_type);

  static constexpr ck::index_t kABBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t kB1BlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t kCShuffleBlockTransferScalarPerVector = 1;
  static constexpr ck::index_t kAcc0BiasTransferSrcScalarPerVector = 1;

  template <
      ck::index_t kGemm1NPerBlock,
      ck::index_t kGemm1NXdlPerWave,
      ck::index_t kCShuffleNXdlPerWavePerShuffle>
  using DeviceOpInstanceTemp = ck::tensor_operation::device::
      DeviceGroupedMultiheadAttentionInfer_Xdl_CShuffle<
          GemmOpConstantsCommon::NumDimG,
          GemmOpConstantsCommon::NumDimM,
          GemmOpConstantsCommon::NumDimN,
          GemmOpConstantsCommon::NumDimK,
          GemmOpConstantsCommon::NumDimO,
          ADataType,
          B0DataType,
          B1DataType,
          CDataType,
          Acc0BiasDataType,
          Acc1BiasDataType,
          AccDataType,
          CShuffleDataType,
          AElementOp,
          B0ElementOp,
          Acc0ElementOp,
          B1ElementOp,
          CElementOp,
          GemmSpec,
          GemmOpConstantsCommon::TensorSpecA,
          GemmOpConstantsCommon::TensorSpecB0,
          GemmOpConstantsCommon::TensorSpecB1,
          GemmOpConstantsCommon::TensorSpecC,
          1,
          GemmOpConstantsGroupedInfer::BlockSize,
          GemmOpConstantsGroupedInfer::MPerBlock,
          GemmOpConstantsGroupedInfer::NPerBlock,
          GemmOpConstantsGroupedInfer::KPerBlock,
          kGemm1NPerBlock,
          GemmOpConstantsGroupedInfer::Gemm1KPerBlock,
          GemmOpConstantsGroupedInfer::AK1,
          GemmOpConstantsGroupedInfer::BK1,
          GemmOpConstantsGroupedInfer::B1K1,
          GemmOpConstantsGroupedInfer::MPerXDL,
          GemmOpConstantsGroupedInfer::NPerXDL,
          GemmOpConstantsGroupedInfer::MXdlPerWave,
          GemmOpConstantsGroupedInfer::NXdlPerWave,
          kGemm1NXdlPerWave,
          GemmOpConstantsGroupedInfer::
              ABlockTransferThreadClusterLengths_AK0_M_AK1,
          GemmOpConstantsGroupedInfer::ABlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedInfer::ABlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedInfer::ABlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector, // TUNABLE
          GemmOpConstantsGroupedInfer::ABlockTransferDstScalarPerVector_AK1,
          GemmOpConstantsGroupedInfer::ABlockLdsExtraM,
          GemmOpConstantsGroupedInfer::
              BBlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsGroupedInfer::BBlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedInfer::BBlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedInfer::BBlockTransferSrcVectorDim,
          kABBlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedInfer::BBlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsGroupedInfer::BBlockLdsExtraN,
          kAcc0BiasTransferSrcScalarPerVector,
          GemmOpConstantsGroupedInfer::
              B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          GemmOpConstantsGroupedInfer::B1BlockTransferThreadClusterArrangeOrder,
          GemmOpConstantsGroupedInfer::B1BlockTransferSrcAccessOrder,
          GemmOpConstantsGroupedInfer::B1BlockTransferSrcVectorDim,
          kB1BlockTransferSrcScalarPerVector,
          GemmOpConstantsGroupedInfer::B1BlockTransferDstScalarPerVector_BK1,
          GemmOpConstantsGroupedInfer::B1BlockLdsExtraN,
          GemmOpConstantsGroupedInfer::CShuffleMXdlPerWavePerShuffle,
          kCShuffleNXdlPerWavePerShuffle,
          GemmOpConstantsGroupedInfer::
              CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          kCShuffleBlockTransferScalarPerVector,
          MaskingSpec>;

  static void Run(GroupedForwardParams& param, hipStream_t stream) {
    if (param.K <= 32 && param.Kv <= 32) {
      constexpr ck::index_t kGemm1NPerBlock = 32;
      constexpr ck::index_t kGemm1NXdlPerWave = 1;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 1;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    } else if (param.K <= 64 && param.Kv <= 64) {
      constexpr ck::index_t kGemm1NPerBlock = 64;
      constexpr ck::index_t kGemm1NXdlPerWave = 2;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 2;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    } else {
      constexpr ck::index_t kGemm1NPerBlock = 128;
      constexpr ck::index_t kGemm1NXdlPerWave = 4;
      constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = 4;

      using DeviceOpInstance = DeviceOpInstanceTemp<
          kGemm1NPerBlock,
          kGemm1NXdlPerWave,
          kCShuffleNXdlPerWavePerShuffle>;

      RunWithDeviceOp<DeviceOpInstance>(param, stream);
    };
  };

  template <typename DeviceOpInstance>
  static void RunWithDeviceOp(GroupedForwardParams& param, hipStream_t stream) {
    std::vector<typename DeviceOpInstance::ProblemDesc> problem_descs;

    for (std::size_t i = 0; i < param.num_batches; i++) {
      int M = param.host_seqstart_q[i + 1] - param.host_seqstart_q[i];
      int N = param.host_seqlen_k.empty()
          ? param.host_seqstart_k[i + 1] - param.host_seqstart_k[i]
          : param.host_seqlen_k[i];
      int K = param.K;
      int Kv = param.Kv;
      int G1 = param.num_heads;

      std::vector<ck::index_t> a_gs_ms_ks_lengths{1, G1, M, K};
      std::vector<ck::index_t> a_gs_ms_ks_strides{
          0, param.q_strides[1], param.q_strides[0], param.q_strides[2]};

      std::vector<ck::index_t> b0_gs_ns_ks_lengths{1, G1, N, K};
      std::vector<ck::index_t> b0_gs_ns_ks_strides{
          0, param.k_strides[1], param.k_strides[0], param.k_strides[2]};

      // to be changed to b1_gs_ns_os_lengths
      std::vector<ck::index_t> b1_gs_os_ns_lengths{1, G1, Kv, N};
      std::vector<ck::index_t> b1_gs_os_ns_strides{
          0, param.v_strides[1], param.v_strides[2], param.v_strides[0]};

      std::vector<ck::index_t> c_gs_ms_os_lengths{1, G1, M, Kv};
      std::vector<ck::index_t> c_gs_ms_os_strides{
          0, param.out_strides[1], param.out_strides[0], param.out_strides[2]};

      std::vector<ck::index_t> d_gs_ms_ns_lengths;
      std::vector<ck::index_t> d_gs_ms_ns_strides;

      if constexpr (has_attn_bias) {
        d_gs_ms_ns_lengths = {1, G1, M, N};
        d_gs_ms_ns_strides = {
            0,
            param.attn_bias_strides[0],
            param.attn_bias_strides[1],
            param.attn_bias_strides[2]};

      } else {
        d_gs_ms_ns_lengths = {1, 1, 1, 1};
        d_gs_ms_ns_strides = {0, 0, 0, 0};
      };

      problem_descs.push_back(
          {a_gs_ms_ks_lengths,
           a_gs_ms_ks_strides,
           b0_gs_ns_ks_lengths,
           b0_gs_ns_ks_strides,
           b1_gs_os_ns_lengths,
           b1_gs_os_ns_strides,
           c_gs_ms_os_lengths,
           c_gs_ms_os_strides,
           d_gs_ms_ns_lengths,
           d_gs_ms_ns_strides,
           {}, // acc1_bias_gs_ms_os_lengths
           {}}); // acc1_bias_gs_ms_os_strides
    }

    float alpha = param.scale;

    auto a_element_op = AElementOp{};
    auto b0_element_op = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op = B1ElementOp{};
    auto c_element_op = CElementOp{};

    auto op = DeviceOpInstance{};
    auto invoker = op.MakeInvoker();

    auto arg_ptr = op.MakeArgumentPointer(
        param.q_ptrs,
        param.k_ptrs,
        param.v_ptrs,
        param.out_ptrs,
        param.attn_bias_ptrs,
        {}, // p_acc1_biases
        problem_descs,
        a_element_op,
        b0_element_op,
        acc0_element_op,
        b1_element_op,
        c_element_op);

    auto sizeInBytes = op.GetWorkSpaceSize(arg_ptr.get());

    SimpleDeviceMem workspace(sizeInBytes);

    op.SetWorkSpacePointer(arg_ptr.get(), workspace.GetDeviceBuffer());

    if (!op.IsSupportedArgument(arg_ptr.get())) {
      std::ostringstream ostr;

      ostr << op.GetTypeString() << " does not support this problem";

      throw std::runtime_error(ostr.str());
    }

    (void)invoker.Run(arg_ptr.get(), StreamConfig{stream, false});
  };
};
