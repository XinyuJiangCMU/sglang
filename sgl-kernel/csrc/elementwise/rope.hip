// HIP implementation of apply_rope_pos_ids_cos_sin_cache
// Adapted from pos_enc.hip with the FlashInfer-style interface
// (pos_ids + cos_sin_cache instead of pre-indexed cache_ptr)

#include "hip/hip_runtime.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "utils_hip.h"

// Vectorized RoPE kernel with cos_sin_cache indexed by pos_ids
// cos_sin_cache layout: [max_seq_len, rotary_dim] where first half is cos, second is sin
template <typename scalar_t, bool INTERLEAVE, bool SAVE_KV_CACHE>
__global__ void apply_rope_pos_ids_kernel(
    scalar_t* __restrict__ q,           // [nnz, H_Q, D]
    scalar_t* __restrict__ k,           // [nnz, H_K, D]
    const scalar_t* __restrict__ v,     // [nnz, H_V, D] or nullptr
    scalar_t* __restrict__ q_rope,      // [nnz, H_Q, D]
    scalar_t* __restrict__ k_rope,      // [nnz, H_K, D]
    scalar_t* __restrict__ k_buffer,    // KV cache k buffer or nullptr
    scalar_t* __restrict__ v_buffer,    // KV cache v buffer or nullptr
    const float* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim]
    const int64_t* __restrict__ pos_ids,      // [nnz]
    const int64_t* __restrict__ kv_cache_loc, // [nnz] or nullptr
    const int nnz,
    const int num_qo_heads,
    const int num_kv_heads,
    const int rotary_dim,
    const int head_dim,
    const size_t q_stride_n,
    const size_t q_stride_h,
    const size_t k_stride_n,
    const size_t k_stride_h,
    const size_t v_stride_n,
    const size_t v_stride_h,
    const size_t q_rope_stride_n,
    const size_t q_rope_stride_h,
    const size_t k_rope_stride_n,
    const size_t k_rope_stride_h,
    const size_t k_buffer_stride_n,
    const size_t k_buffer_stride_h,
    const size_t v_buffer_stride_n,
    const size_t v_buffer_stride_h) {
  // Each block handles one token
  const int token_idx = blockIdx.x;
  if (token_idx >= nnz) return;

  const int64_t pos = pos_ids[token_idx];
  const int half_rotary = rotary_dim / 2;

  // cos_sin_cache: [max_pos, rotary_dim] - first half cos, second half sin
  const float* cos_ptr = cos_sin_cache + pos * rotary_dim;
  const float* sin_ptr = cos_ptr + half_rotary;

  // Apply RoPE to query heads
  for (int i = threadIdx.x; i < num_qo_heads * half_rotary; i += blockDim.x) {
    const int head_idx = i / half_rotary;
    const int rot_offset = i % half_rotary;

    int x_index, y_index;
    if constexpr (!INTERLEAVE) {
      // NeoX style
      x_index = rot_offset;
      y_index = half_rotary + rot_offset;
    } else {
      // GPT-J style
      x_index = 2 * rot_offset;
      y_index = 2 * rot_offset + 1;
    }

    float cos_val = cos_ptr[rot_offset];
    float sin_val = sin_ptr[rot_offset];

    scalar_t* q_ptr = q + token_idx * q_stride_n + head_idx * q_stride_h;
    scalar_t* q_rope_ptr = q_rope + token_idx * q_rope_stride_n + head_idx * q_rope_stride_h;

    float x = static_cast<float>(q_ptr[x_index]);
    float y = static_cast<float>(q_ptr[y_index]);
    q_rope_ptr[x_index] = static_cast<scalar_t>(x * cos_val - y * sin_val);
    q_rope_ptr[y_index] = static_cast<scalar_t>(y * cos_val + x * sin_val);

    // Copy non-rotary dimensions
    if (rot_offset == 0) {
      for (int d = rotary_dim; d < head_dim; d++) {
        q_rope_ptr[d] = q_ptr[d];
      }
    }
  }

  // Apply RoPE to key heads
  for (int i = threadIdx.x; i < num_kv_heads * half_rotary; i += blockDim.x) {
    const int head_idx = i / half_rotary;
    const int rot_offset = i % half_rotary;

    int x_index, y_index;
    if constexpr (!INTERLEAVE) {
      x_index = rot_offset;
      y_index = half_rotary + rot_offset;
    } else {
      x_index = 2 * rot_offset;
      y_index = 2 * rot_offset + 1;
    }

    float cos_val = cos_ptr[rot_offset];
    float sin_val = sin_ptr[rot_offset];

    scalar_t* k_ptr = k + token_idx * k_stride_n + head_idx * k_stride_h;
    scalar_t* k_rope_ptr = k_rope + token_idx * k_rope_stride_n + head_idx * k_rope_stride_h;

    float x = static_cast<float>(k_ptr[x_index]);
    float y = static_cast<float>(k_ptr[y_index]);
    k_rope_ptr[x_index] = static_cast<scalar_t>(x * cos_val - y * sin_val);
    k_rope_ptr[y_index] = static_cast<scalar_t>(y * cos_val + x * sin_val);

    // Copy non-rotary dimensions
    if (rot_offset == 0) {
      for (int d = rotary_dim; d < head_dim; d++) {
        k_rope_ptr[d] = k_ptr[d];
      }
    }
  }

  // Optionally save KV cache
  if constexpr (SAVE_KV_CACHE) {
    const int64_t cache_loc = kv_cache_loc[token_idx];
    // Save k_rope to k_buffer
    for (int i = threadIdx.x; i < num_kv_heads * head_dim; i += blockDim.x) {
      const int head_idx = i / head_dim;
      const int d = i % head_dim;
      scalar_t* k_rope_ptr = k_rope + token_idx * k_rope_stride_n + head_idx * k_rope_stride_h;
      scalar_t* kb_ptr = k_buffer + cache_loc * k_buffer_stride_n + head_idx * k_buffer_stride_h;
      kb_ptr[d] = k_rope_ptr[d];
    }
    // Save v to v_buffer
    for (int i = threadIdx.x; i < num_kv_heads * head_dim; i += blockDim.x) {
      const int head_idx = i / head_dim;
      const int d = i % head_dim;
      const scalar_t* v_ptr = v + token_idx * v_stride_n + head_idx * v_stride_h;
      scalar_t* vb_ptr = v_buffer + cache_loc * v_buffer_stride_n + head_idx * v_buffer_stride_h;
      vb_ptr[d] = v_ptr[d];
    }
  }
}

void apply_rope_pos_ids_cos_sin_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    bool enable_pdl,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc) {
  // PDL is not supported on HIP
  (void)enable_pdl;

  const bool save_kv_cache = v.has_value();

  unsigned int nnz = q.size(0);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int rotary_dim = cos_sin_cache.size(1);

  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  size_t v_stride_n = save_kv_cache ? v->stride(0) : 0;
  size_t v_stride_h = save_kv_cache ? v->stride(1) : 0;
  size_t k_buffer_stride_n = save_kv_cache ? k_buffer->stride(0) : 0;
  size_t k_buffer_stride_h = save_kv_cache ? k_buffer->stride(1) : 0;
  size_t v_buffer_stride_n = save_kv_cache ? v_buffer->stride(0) : 0;
  size_t v_buffer_stride_h = save_kv_cache ? v_buffer->stride(1) : 0;
  auto kv_cache_loc_ptr = save_kv_cache ? kv_cache_loc->data_ptr<int64_t>() : nullptr;

  dim3 grid(nnz);
  int max_work = std::max(num_qo_heads, num_kv_heads) * rotary_dim / 2;
  dim3 block(std::min(max_work, 512));
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  // Dispatch by dtype
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      q.scalar_type(), "apply_rope_pos_ids", [&] {
    if (save_kv_cache) {
      if (interleave) {
        hipLaunchKernelGGL((apply_rope_pos_ids_kernel<scalar_t, true, true>),
            grid, block, 0, stream,
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v->data_ptr<scalar_t>(),
            q_rope.data_ptr<scalar_t>(), k_rope.data_ptr<scalar_t>(),
            k_buffer->data_ptr<scalar_t>(), v_buffer->data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<float>(), pos_ids.data_ptr<int64_t>(),
            kv_cache_loc_ptr,
            nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            v_stride_n, v_stride_h,
            q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
            k_buffer_stride_n, k_buffer_stride_h, v_buffer_stride_n, v_buffer_stride_h);
      } else {
        hipLaunchKernelGGL((apply_rope_pos_ids_kernel<scalar_t, false, true>),
            grid, block, 0, stream,
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v->data_ptr<scalar_t>(),
            q_rope.data_ptr<scalar_t>(), k_rope.data_ptr<scalar_t>(),
            k_buffer->data_ptr<scalar_t>(), v_buffer->data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<float>(), pos_ids.data_ptr<int64_t>(),
            kv_cache_loc_ptr,
            nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            v_stride_n, v_stride_h,
            q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
            k_buffer_stride_n, k_buffer_stride_h, v_buffer_stride_n, v_buffer_stride_h);
      }
    } else {
      if (interleave) {
        hipLaunchKernelGGL((apply_rope_pos_ids_kernel<scalar_t, true, false>),
            grid, block, 0, stream,
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            nullptr,
            q_rope.data_ptr<scalar_t>(), k_rope.data_ptr<scalar_t>(),
            nullptr, nullptr,
            cos_sin_cache.data_ptr<float>(), pos_ids.data_ptr<int64_t>(),
            nullptr,
            nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            0, 0,
            q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
            0, 0, 0, 0);
      } else {
        hipLaunchKernelGGL((apply_rope_pos_ids_kernel<scalar_t, false, false>),
            grid, block, 0, stream,
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            nullptr,
            q_rope.data_ptr<scalar_t>(), k_rope.data_ptr<scalar_t>(),
            nullptr, nullptr,
            cos_sin_cache.data_ptr<float>(), pos_ids.data_ptr<int64_t>(),
            nullptr,
            nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            0, 0,
            q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
            0, 0, 0, 0);
      }
    }
  });
}
