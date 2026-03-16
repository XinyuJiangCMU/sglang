// HIP implementation of speculative sampling kernels for AMD GPUs.
// Ports tree_speculative_sampling_target_only from the CUDA/FlashInfer version,
// and adds top_k_renorm_probs / top_p_renorm_probs using ATen ops.

#include "hip/hip_runtime.h"

#include <ATen/cuda/CUDAContext.h>
#include <hipcub/hipcub.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "pytorch_extension_utils_rocm.h"

// ---------------------------------------------------------------------------
// Utility: vectorized load/store (replaces flashinfer::vec_t)
// ---------------------------------------------------------------------------
template <typename T, uint32_t VEC_SIZE>
struct vec_t {
  T data[VEC_SIZE];

  __device__ __forceinline__ void load(const T* ptr) {
    if constexpr (VEC_SIZE * sizeof(T) == 16) {
      *reinterpret_cast<float4*>(data) = *reinterpret_cast<const float4*>(ptr);
    } else if constexpr (VEC_SIZE * sizeof(T) == 8) {
      *reinterpret_cast<float2*>(data) = *reinterpret_cast<const float2*>(ptr);
    } else if constexpr (VEC_SIZE * sizeof(T) == 4) {
      *reinterpret_cast<float*>(data) = *reinterpret_cast<const float*>(ptr);
    } else {
      for (uint32_t i = 0; i < VEC_SIZE; ++i) data[i] = ptr[i];
    }
  }

  __device__ __forceinline__ void fill(T val) {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) data[i] = val;
  }

  __device__ __forceinline__ T& operator[](uint32_t i) { return data[i]; }
  __device__ __forceinline__ const T& operator[](uint32_t i) const { return data[i]; }
};

// ---------------------------------------------------------------------------
// Shared memory layout for TreeSpeculativeSamplingTargetOnly
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_THREADS>
struct SamplingTempStorage {
  union {
    typename hipcub::BlockReduce<float, BLOCK_THREADS>::TempStorage reduce;
    typename hipcub::BlockScan<float, BLOCK_THREADS>::TempStorage scan;
  } block_prim;
  struct {
    float value;
  } block_aggregate;
  uint32_t sampled_id;
};

// ---------------------------------------------------------------------------
// ceil_div utility
// ---------------------------------------------------------------------------
__device__ __host__ __forceinline__ uint32_t ceil_div(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Kernel: TreeSpeculativeSamplingTargetOnly
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType, typename IdType2>
__global__ void TreeSpeculativeSamplingTargetOnlyKernel(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {

  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS>&>(smem_sampling);

  // --- Phase 1: Sequential tree walk (single-threaded per block) ---
  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      DType target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      if (coin <= prob_acc / threshold_acc ||
          target_prob_single >= threshold_single) {
        // accept token
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] =
            draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] =
            target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;

  // --- Phase 2: Parallel sampling from relu(target - draft) ---
  coin = uniform_samples_for_final_sampling[bx];

  // Pass 1: compute sum of relu(q - p) via BlockReduce
  DType sum_relu_q_minus_p(0);
  vec_t<DType, VEC_SIZE> q_vec, p_vec;
  DType relu_q_minus_p[VEC_SIZE];
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset +
                 i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        p_vec.load(draft_probs + cur_prob_offset +
                   i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], DType(0));
    }
    sum_relu_q_minus_p +=
        hipcub::BlockReduce<DType, BLOCK_THREADS>(temp_storage.block_prim.reduce)
            .Sum(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }
  temp_storage.sampled_id = d - 1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  DType u = coin * sum_relu_q_minus_p;

  // Pass 2: find the sampled token via prefix scan
  DType aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset +
                 i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        p_vec.load(draft_probs + cur_prob_offset +
                   i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    DType thread_vals[VEC_SIZE];
    DType thread_sum = DType(0);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_vals[j] = max(q_vec[j] - p_vec[j], DType(0));
      thread_sum += thread_vals[j];
    }

    // Inclusive prefix sum across the block
    DType thread_inclusive;
    hipcub::BlockScan<DType, BLOCK_THREADS>(temp_storage.block_prim.scan)
        .InclusiveSum(thread_sum, thread_inclusive);
    __syncthreads();

    // Per-thread exclusive prefix = inclusive - own sum, plus global aggregate
    DType thread_exclusive = aggregate_relu_q_minus_p + thread_inclusive - thread_sum;
    DType running = thread_exclusive;

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      running += thread_vals[j];
      if (running > u) {
        uint32_t pos = i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE + j;
        if (pos < d) {
          atomicMin(&temp_storage.sampled_id, pos);
        }
        break;
      }
    }
    __syncthreads();

    // Block total = last thread's inclusive sum; broadcast via shared memory
    if (tx == BLOCK_THREADS - 1) {
      temp_storage.block_aggregate.value = thread_inclusive;
    }
    __syncthreads();
    aggregate_relu_q_minus_p += temp_storage.block_aggregate.value;

    // Early exit if sample found or aggregate exceeds threshold
    if (temp_storage.sampled_id < d - 1 || aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  predicts[last_accepted_retrive_idx] = temp_storage.sampled_id;
}

// ---------------------------------------------------------------------------
// Host wrapper for TreeSpeculativeSamplingTargetOnly
// ---------------------------------------------------------------------------
template <typename DType, typename IdType, typename IdType2>
hipError_t LaunchTreeSpeculativeSamplingTargetOnly(
    IdType* predicts, IdType* accept_index, IdType* accept_token_num,
    IdType2* candidates, IdType2* retrive_index, IdType2* retrive_next_token,
    IdType2* retrive_next_sibling, DType* uniform_samples,
    DType* uniform_samples_for_final_sampling, DType* target_probs,
    DType* draft_probs, uint32_t batch_size, uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens, uint32_t d, DType threshold_single,
    DType threshold_acc, bool deterministic, hipStream_t stream) {

  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16u / (uint32_t)sizeof(DType), d);
  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS>);

  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);

  DType capped_threshold_acc = fmaxf(threshold_acc, 1e-9f);

  // Dispatch based on vec_size (common values for float32 vocab sizes)
#define LAUNCH_KERNEL(VS)                                                      \
  do {                                                                         \
    auto kernel =                                                              \
        TreeSpeculativeSamplingTargetOnlyKernel<BLOCK_THREADS, VS, true,       \
                                                DType, IdType, IdType2>;       \
    hipError_t err = hipFuncSetAttribute(                                      \
        (const void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize,       \
        smem_size);                                                            \
    if (err != hipSuccess) return err;                                         \
    hipLaunchKernelGGL(kernel, nblks, nthrs, smem_size, stream, predicts,      \
                       accept_index, accept_token_num, candidates,             \
                       retrive_index, retrive_next_token,                      \
                       retrive_next_sibling, uniform_samples,                  \
                       uniform_samples_for_final_sampling, target_probs,       \
                       draft_probs, batch_size, num_speculative_tokens,        \
                       num_draft_tokens, d, threshold_single,                  \
                       capped_threshold_acc);                                  \
    return hipGetLastError();                                                  \
  } while (0)

  switch (vec_size) {
    case 4:
      LAUNCH_KERNEL(4);
    case 2:
      LAUNCH_KERNEL(2);
    default:
      LAUNCH_KERNEL(1);
  }
#undef LAUNCH_KERNEL
}

// ---------------------------------------------------------------------------
// PyTorch interface: tree_speculative_sampling_target_only
// ---------------------------------------------------------------------------
void tree_speculative_sampling_target_only(
    at::Tensor predicts, at::Tensor accept_index,
    at::Tensor accept_token_num, at::Tensor candidates,
    at::Tensor retrive_index, at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling, at::Tensor uniform_samples,
    at::Tensor uniform_samples_for_final_sampling, at::Tensor target_probs,
    at::Tensor draft_probs, double threshold_single, double threshold_acc,
    bool deterministic = true) {

  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(uniform_samples_for_final_sampling);
  CHECK_INPUT(target_probs);
  CHECK_INPUT(draft_probs);
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, uniform_samples);
  CHECK_DIM(3, target_probs);
  CHECK_DIM(3, draft_probs);
  TORCH_CHECK(predicts.scalar_type() == at::kInt, "predicts must be int32");
  TORCH_CHECK(accept_index.scalar_type() == at::kInt, "accept_index must be int32");
  TORCH_CHECK(accept_token_num.scalar_type() == at::kInt, "accept_token_num must be int32");
  TORCH_CHECK(candidates.scalar_type() == at::kLong, "candidates must be int64");
  TORCH_CHECK(retrive_index.scalar_type() == at::kLong, "retrive_index must be int64");
  TORCH_CHECK(retrive_next_token.scalar_type() == at::kLong, "retrive_next_token must be int64");
  TORCH_CHECK(retrive_next_sibling.scalar_type() == at::kLong, "retrive_next_sibling must be int64");
  TORCH_CHECK(uniform_samples.scalar_type() == at::kFloat, "uniform_samples must be float32");
  TORCH_CHECK(uniform_samples_for_final_sampling.scalar_type() == at::kFloat, "uniform_samples_for_final_sampling must be float32");
  TORCH_CHECK(target_probs.scalar_type() == at::kFloat, "target_probs must be float32");
  TORCH_CHECK(draft_probs.scalar_type() == at::kFloat, "draft_probs must be float32");

  unsigned int batch_size = uniform_samples.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_probs.size(2);

  hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  hipError_t status =
      LaunchTreeSpeculativeSamplingTargetOnly<float, int32_t, int64_t>(
          static_cast<int32_t*>(predicts.data_ptr()),
          static_cast<int32_t*>(accept_index.data_ptr()),
          static_cast<int32_t*>(accept_token_num.data_ptr()),
          static_cast<int64_t*>(candidates.data_ptr()),
          static_cast<int64_t*>(retrive_index.data_ptr()),
          static_cast<int64_t*>(retrive_next_token.data_ptr()),
          static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
          static_cast<float*>(uniform_samples.data_ptr()),
          static_cast<float*>(uniform_samples_for_final_sampling.data_ptr()),
          static_cast<float*>(target_probs.data_ptr()),
          static_cast<float*>(draft_probs.data_ptr()), batch_size,
          num_spec_step, num_draft_tokens, vocab_size,
          static_cast<float>(threshold_single),
          static_cast<float>(threshold_acc), deterministic, stream);

  TORCH_CHECK(status == hipSuccess,
              "TreeSpeculativeSamplingTargetOnly failed: ",
              hipGetErrorString(status));
}

// ---------------------------------------------------------------------------
// PyTorch interface: top_k_renorm_probs (ATen wrapper)
// ---------------------------------------------------------------------------
void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr,
                        int64_t top_k_val) {
  CHECK_INPUT(probs);

  int64_t batch_size = probs.size(0);
  int64_t vocab_size = probs.size(1);

  // Determine per-row top_k values
  at::Tensor top_k_arr;
  if (maybe_top_k_arr.has_value()) {
    top_k_arr = maybe_top_k_arr.value().to(torch::kLong);
  } else {
    top_k_arr = torch::full({batch_size}, top_k_val, torch::dtype(torch::kLong).device(probs.device()));
  }

  // Clamp to valid range
  top_k_arr = top_k_arr.clamp(1, vocab_size);
  int64_t max_k = top_k_arr.max().item<int64_t>();

  if (max_k >= vocab_size) {
    renorm_probs.copy_(probs);
    return;
  }

  // Get top-k values per row
  auto [topk_vals, topk_indices] =
      probs.topk(max_k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);

  // Per-row threshold: value at position top_k[i]-1
  auto ks = (top_k_arr - 1).clamp(0, max_k - 1);  // (batch,)
  auto thresholds = topk_vals.gather(
      1, ks.unsqueeze(-1));  // (batch, 1)

  // Mask and renormalize
  auto filtered = probs.clone();
  filtered.masked_fill_(probs < thresholds, 0.0f);
  auto sums = filtered.sum(-1, /*keepdim=*/true).clamp_min(1e-10f);
  renorm_probs.copy_(filtered / sums);
}

// ---------------------------------------------------------------------------
// PyTorch interface: top_p_renorm_probs (ATen wrapper)
// ---------------------------------------------------------------------------
void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr,
                        double top_p_val) {
  CHECK_INPUT(probs);

  int64_t batch_size = probs.size(0);

  // Determine per-row top_p values
  at::Tensor top_p_arr;
  if (maybe_top_p_arr.has_value()) {
    top_p_arr = maybe_top_p_arr.value().to(probs.dtype());
  } else {
    top_p_arr = torch::full({batch_size}, top_p_val,
                            torch::dtype(probs.dtype()).device(probs.device()));
  }

  // Sort descending
  auto [sorted_probs, sorted_indices] =
      probs.sort(/*dim=*/-1, /*descending=*/true);
  auto cumsum = sorted_probs.cumsum(-1);

  // Shift cumsum right so the token pushing over threshold is included
  auto cumsum_shifted = cumsum - sorted_probs;
  auto mask = cumsum_shifted < top_p_arr.unsqueeze(-1);
  // Always keep at least the top token
  mask.index_put_({torch::indexing::Slice(), 0}, true);

  // Zero out filtered tokens
  sorted_probs.masked_fill_(~mask, 0.0f);

  // Scatter back to original order
  auto filtered = torch::zeros_like(probs);
  filtered.scatter_(1, sorted_indices, sorted_probs);

  // Renormalize
  auto sums = filtered.sum(-1, /*keepdim=*/true).clamp_min(1e-10f);
  renorm_probs.copy_(filtered / sums);
}
