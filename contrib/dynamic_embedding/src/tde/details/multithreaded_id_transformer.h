#pragma once
#include <libcuckoo/cuckoohash_map.hh>
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

template <typename T = uint32_t>
struct MTBitmap {
  explicit MTBitmap(int64_t num_bits);
  MTBitmap(const MTBitmap&) = delete;
  MTBitmap(MTBitmap&&) noexcept = default;

  int64_t NextFreeBit();
  void FreeBit(int64_t offset);
  bool Full() const;

  static constexpr int64_t num_bits_per_value = sizeof(T) * 8;

  mutable std::mutex mu_;
  const int64_t num_total_bits_;
  const int64_t num_values_;
  std::unique_ptr<T[]> values_;

  int64_t next_free_bit_;
};

/**
 * MultithreadedIDTransformer
 *
 * Transform GlobalID to CacheID by multithreaded flat hash map
 * @tparam LXURecord The extension type used for eviction strategy.
 * @tparam MTBitmap The bitmap class to record the free cache ids.
 */
template <typename LXURecord, typename MTBitmap = MTBitmap<uint32_t>>
class MultithreadedIDTransformer {
 public:
  using lxu_record_t = LXURecord;
  using record_t = TransformerRecord<lxu_record_t>;
  static constexpr std::string_view type_ = "multithreaded";

  explicit MultithreadedIDTransformer(int64_t num_embedding);
  MultithreadedIDTransformer(const MultithreadedIDTransformer<LXURecord, MTBitmap>&) = delete;
  MultithreadedIDTransformer(MultithreadedIDTransformer<LXURecord, MTBitmap>&&) noexcept =
      default;

  static MultithreadedIDTransformer<LXURecord, MTBitmap> Create(
      int64_t num_embedding,
      const nlohmann::json& json) {
    return MultithreadedIDTransformer<LXURecord, MTBitmap>(num_embedding);
  }

  /**
   * Transform global ids to cache ids
   *
   * @tparam Update Update the eviction strategy tag type. Update LXU Record
   * @tparam Fetch Fetch the not existing global-id/cache-id pair. It is used
   * by dynamic embedding parameter server.
   *
   * @param global_ids Global ID vector
   * @param cache_ids [out] Cache ID vector
   * @param update update lambda. See `Update` doc.
   * @param fetch fetch lambda. See `Fetch` doc.
   * @return true if all transformed, otherwise need eviction.
   */
  template <
      typename Update = decltype(transform_default::NoUpdate<LXURecord>),
      typename Fetch = decltype(transform_default::NoFetch)>
  bool Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Update update = transform_default::NoUpdate<LXURecord>,
      Fetch fetch = transform_default::NoFetch);

  void Evict(tcb::span<const int64_t> global_ids);

  MoveOnlyFunction<std::optional<record_t>()> Iterator() const;

 private:
  struct CacheValue {
    int64_t cache_id_;
    LXURecord lxu_record_;

    CacheValue() = default;

    CacheValue(int64_t cache_id, LXURecord lxu_record)
      : cache_id_(cache_id), lxu_record_(lxu_record) {}

    template <typename Creator>
    CacheValue(Creator creator) {
      CacheValue other = creator();
      cache_id_ = other.cache_id_;
      lxu_record_ = other.lxu_record_;
    }
  };

  libcuckoo::cuckoohash_map<int64_t, CacheValue> global_id2cache_value_;

  MTBitmap bitmap_;
  int64_t num_threads_;
};

} // namespace tde::details

#include "tde/details/multithreaded_id_transformer_impl.h"
