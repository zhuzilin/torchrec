#include "tde/dense_ps.h"

namespace tde {

DensePS::DensePS(std::string table_name, const std::string& io_config)
    : table_name_(std::move(table_name)) {
  auto pos = io_config.find(details::k_schema_separator);
  TORCH_CHECK(
      pos != std::string::npos,
      "config string should be schema://cfg_string, cannot find schema");

  std::string schema = io_config.substr(0, pos);
  std::string rest_cfg = io_config.substr(pos + details::k_schema_separator.size());
  auto& reg = details::IORegistry::Instance();
  provider_ = reg.Resolve(schema);
  instance_ = provider_.Initialize(rest_cfg.c_str());
}

DensePS::~DensePS() {
  if (instance_ == nullptr)
    return;
  
  provider_.Finalize(instance_);
}

std::vector<uint64_t> CreateKeyOffsets(StringList keys) {
  std::vector<uint64_t> key_offsets_(keys.size() + 1);
  key_offsets_[0] = 0;
  for (uint32_t i = 0; i < keys.size(); ++i) {
    key_offsets_[i + 1] = key_offsets_[i] + keys[i].size();
  }
  return key_offsets_;
}

std::string CreateConcatKey(StringList keys, const std::vector<uint64_t>& offsets) {
  TORCH_CHECK(keys.size() + 1 == offsets.size());
  std::string concat_key(offsets.back(), '\0');
  for (uint32_t i = 0; i < keys.size(); ++i) {
    memcpy(concat_key.data() + offsets[i], keys[i].data(), keys[i].size());
  }
  return concat_key;
}

std::vector<uint64_t> CreateTensorOffsets(TensorList tensors) {
  std::vector<uint64_t> tensor_offsets(tensors.size() + 1);
  tensor_offsets[0] = 0;
  for (uint32_t i = 0; i < tensors.size(); ++i) {
    tensor_offsets[i + 1] = tensor_offsets[i] + tensors[i].numel() * tensors[i].element_size();
  }
  return tensor_offsets;
}

torch::Tensor CreateConcatTensor(TensorList tensors) {
  std::vector<torch::Tensor> flatten_tensors;
  for (uint32_t i = 0; i < tensors.size(); ++i) {
    flatten_tensors.push_back(tensors[i].flatten());
  }
  torch::Tensor concat_tensor = torch::cat(flatten_tensors, 0);
  return concat_tensor.cpu();
}

struct DenseLoadContext {
  std::unique_ptr<Notification> notifcation_;
  TensorList tensors_;
};

static void OnTensorLoaded(
    void* ctx,
    uint32_t offset,
    void* data,
    uint32_t data_len) {
  if (data_len == 0)
    return;
  DenseLoadContext* c = static_cast<DenseLoadContext*>(ctx);
  TORCH_CHECK(offset < c->tensors_.size());
  // TODO: support tensor other than float.
  TORCH_CHECK(data_len == c->tensors_[offset].numel() * sizeof(float));
  torch::Tensor tensor = at::from_blob(data, c->tensors_[offset].sizes());
  c->tensors_[offset].copy_(tensor);
}

void DensePS::Load(c10::intrusive_ptr<StringList> keys, c10::intrusive_ptr<TensorList> tensors) {
  uint32_t num_tensors = keys->size();
  std::vector<uint64_t> key_offsets = CreateKeyOffsets(*keys);
  std::string concat_key = CreateConcatKey(*keys, key_offsets);

  std::vector<uint64_t> tensor_offsets = CreateTensorOffsets(*tensors);

  DenseLoadContext ctx{
    .notifcation_ = std::make_unique<Notification>(),
    .tensors_ = *tensors,
  };

  provider_.DenseLoad(instance_, details::IODenseLoadParameter{
    .table_name_ = table_name_.c_str(),
    .num_tensors_ = num_tensors,
    .keys_ = concat_key.c_str(),
    .key_offsets_ = key_offsets.data(),
    .tensor_offsets_ = tensor_offsets.data(),
    .on_complete_context_ = &ctx,
    .on_tensor_loaded_ = OnTensorLoaded,
    .on_all_loaded_ = +[](void* ctx) {
      DenseLoadContext* c = static_cast<DenseLoadContext*>(ctx);
      c->notifcation_->Done();
    }
  });
  ctx.notifcation_->Wait();
}

void DensePS::Save(c10::intrusive_ptr<StringList> keys, c10::intrusive_ptr<TensorList> tensors) {
  uint32_t num_tensors = keys->size();
  std::vector<uint64_t> key_offsets = CreateKeyOffsets(*keys);
  std::string concat_key = CreateConcatKey(*keys, key_offsets);

  std::vector<uint64_t> tensor_offsets = CreateTensorOffsets(*tensors);
  torch::Tensor concat_tensor = CreateConcatTensor(*tensors);

  Notification notification;

  provider_.DenseSave(instance_, details::IODenseSaveParameter{
    .table_name_ = table_name_.c_str(),
    .num_tensors_ = num_tensors,
    .keys_ = concat_key.c_str(),
    .key_offsets_ = key_offsets.data(),
    .data_ = concat_tensor.data_ptr<float>(),
    .tensor_offsets_ = tensor_offsets.data(),
    .on_complete_context_ = &notification,
    .on_save_complete_ = +[](void* ctx) {
      Notification* notification = static_cast<Notification*>(ctx);
      notification->Done();
    }
  });
  notification.Wait();
}

} // namespace tde
