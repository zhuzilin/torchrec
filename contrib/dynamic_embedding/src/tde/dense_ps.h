#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <deque>
#include <utility>
#include "tde/details/io.h"
#include "tde/notification.h"
#include "tde/tensor_list.h"
#include "tde/string_list.h"

namespace tde {

class DensePS : public torch::CustomClassHolder {
 public:
  DensePS(std::string table_name,
          const std::string& io_config);
  ~DensePS();

  void Load(c10::intrusive_ptr<StringList> keys, c10::intrusive_ptr<TensorList> tensors);
  void Save(c10::intrusive_ptr<StringList> keys, c10::intrusive_ptr<TensorList> tensors);

 private:
  std::string table_name_;
  details::IOProvider provider_{};
  void* instance_{};
};

} // namespace tde
