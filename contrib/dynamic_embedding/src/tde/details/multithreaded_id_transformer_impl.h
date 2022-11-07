#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include "tde/details/bits_op.h"

namespace tde::details {

template <typename T>
inline MTBitmap<T>::MTBitmap(int64_t num_bits)
    : num_total_bits_(num_bits),
      num_values_((num_bits + num_bits_per_value - 1) / num_bits_per_value),
      values_(new T[num_values_]),
      next_free_bit_(0) {
  std::fill(values_.get(), values_.get() + num_values_, -1);
}

template <typename T>
inline int64_t MTBitmap<T>::NextFreeBit() {
  std::lock_guard<std::mutex> lock(mu_);
  int64_t result = next_free_bit_;
  int64_t offset = result / num_bits_per_value;
  T value = values_[offset];
  // set the last 1 bit to zero
  values_[offset] = value & (value - 1);
  while (values_[offset] == 0 && offset < num_values_) {
    offset++;
  }
  value = values_[offset];
  if (C10_LIKELY(value)) {
    next_free_bit_ = offset * num_bits_per_value + Ctz(value);
  } else {
    next_free_bit_ = num_total_bits_;
  }

  return result;
}

template <typename T>
inline void MTBitmap<T>::FreeBit(int64_t offset) {
  int64_t mask_offset = offset / num_bits_per_value;
  int64_t bit_offset = offset % num_bits_per_value;
  values_[mask_offset] |= 1 << bit_offset;
  next_free_bit_ = std::min(offset, next_free_bit_);
}
template <typename T>
inline bool MTBitmap<T>::Full() const {
  std::lock_guard<std::mutex> lock(mu_);
  return next_free_bit_ >= num_total_bits_;
}

template <typename LXURecord, typename T>
inline MultithreadedIDTransformer<LXURecord, T>::MultithreadedIDTransformer(
    int64_t num_embedding)
    : bitmap_(num_embedding),
      num_threads_(4) {
  global_id2cache_value_.reserve(num_embedding);
}

template <typename LXURecord, typename T>
template <typename Update, typename Fetch>
inline bool MultithreadedIDTransformer<LXURecord, T>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Update update,
    Fetch fetch) {
  int64_t num_global_ids = global_ids.size();
  std::vector<std::thread> threads;
  for (int64_t t = 0; t < num_threads_; ++t) {
    int64_t start = num_global_ids / num_threads_ * t;
    int64_t end = t == num_threads_ - 1 ?
      num_global_ids - start :
      num_global_ids / num_threads_ * (t + 1);
    threads.emplace_back(std::thread([&, start, end, t] {
      for (size_t i = start; i < end; ++i) {
        int64_t global_id = global_ids[i];
        int64_t& cache_id = cache_ids[i];
        bool is_full = false;
        auto update_fn = [&, this] (CacheValue& cache_value) {
          cache_id = cache_value.cache_id_;
          cache_value.lxu_record_ =
              update(cache_value.lxu_record_, global_id, cache_id);
        };
        auto create_fn = [&, this] () {
          // The transformer is full.
          if (C10_UNLIKELY(bitmap_.Full())) {
            is_full = true;
            return CacheValue();
          }
          auto stored_cache_id = bitmap_.NextFreeBit();
          cache_id = stored_cache_id;
          LXURecord record = update(std::nullopt, global_id, cache_id);
          fetch(global_id, cache_id);
          return CacheValue(stored_cache_id, record);
        };
        global_id2cache_value_.upsert(global_id, std::move(update_fn), CacheValue());
        if (is_full)
          break;
      }
    }));
  }
  for (int64_t t = 0; t < num_threads_; ++t) {
    threads[t].join();
  }
  return true;
}

template <typename LXURecord, typename T>
inline void MultithreadedIDTransformer<LXURecord, T>::Evict(
    tcb::span<const int64_t> global_ids) {
  auto lock_table = global_id2cache_value_.lock_table();
  for (const int64_t global_id : global_ids) {
    auto iter = lock_table.find(global_id);
    if (iter == lock_table.end()) {
      continue;
    }
    int64_t cache_id = iter->second.cache_id_;
    lock_table.erase(iter);
    bitmap_.FreeBit(cache_id);
  }
}

template <typename LXURecord, typename T>
inline auto MultithreadedIDTransformer<LXURecord, T>::Iterator() const
    -> MoveOnlyFunction<std::optional<record_t>()> {
  auto lock_table = global_id2cache_value_.lock_table();
  auto iter = lock_table.begin();
  return [iter, lock_table, this]() mutable -> std::optional<record_t> {
    if (iter != lock_table.end()) {
      auto record = record_t{
          .global_id_ = iter->first,
          .cache_id_ = iter->second.cache_id_,
          .lxu_record_ = iter->second.lxu_record_,
      };
      iter++;
      return record;
    } else {
      return {};
    }
  };
}

} // namespace tde::details
