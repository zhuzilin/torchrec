#pragma once
#include <torch/torch.h>

namespace tde {

class StringList : public torch::CustomClassHolder {
  using Container = std::vector<std::string>;

 public:
  StringList() = default;

  void push_back(std::string tensor) {
    strings_.push_back(tensor);
  }
  int64_t size() const {
    return strings_.size();
  }
  std::string& operator[](int64_t index) {
    return strings_[index];
  }

  Container::const_iterator begin() const {
    return strings_.begin();
  }
  Container::const_iterator end() const {
    return strings_.end();
  }

 private:
  Container strings_;
};

} // namespace tde
