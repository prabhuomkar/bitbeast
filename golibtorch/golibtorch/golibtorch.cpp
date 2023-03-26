#include <torch/torch.h>
#include <torch/script.h>
#include "golibtorch.h"

class Model {
  torch::jit::script::Module model;
public:
  Model(const std::string &modelFile);
  c10::IValue Result(float *inputData, int *channels, int *width, int *height);
};

Model::Model(const std::string &modelFile){
  model = torch::jit::load(modelFile);
}

c10::IValue Model::Result(float *inputData, int *channels, int *width, int *height) {
  std::vector<int64_t> sizes = {1, *channels, *width, *height};
  torch::Tensor input = torch::from_blob(inputData, torch::IntList(sizes));
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(input);
  return model.forward(inputs);
}

mModel NewModel(char *modelFile){
  try {
    const auto model = new Model(modelFile);
    return (void *)model;
  } catch(const std::invalid_argument &ex) {
    return nullptr;
  }
}

Result *GetResult(mModel model, float *inputData, int *channels, int *width, int *height) {
  auto initializedModel = (Model *)model;
  if (initializedModel == nullptr) {
    return nullptr;
  }
  auto result = initializedModel->Result(inputData, channels, width, height);
  auto resultDict = result.toGenericDict();
  std::vector<std::string> labels;
  std::vector<float> scores;
  for (auto it = resultDict.begin(); it != resultDict.end(); ++it) {
    labels.push_back(it->key().toStringRef());
    scores.push_back(it->value().toDouble());
  }
  int numElements = labels.size();
  Result* resultPtr = new Result();
  resultPtr->labels = new char*[numElements];
  for (int i = 0; i < numElements; i++) {
      resultPtr->labels[i] = new char[labels[i].length() + 1];
      std::strcpy(resultPtr->labels[i], labels[i].c_str());
  }
  resultPtr->scores = new float[numElements];
  for (int i = 0; i < numElements; i++) {
      resultPtr->scores[i] = scores[i];
  }
  return resultPtr;
}

void DeleteModel(mModel model) {
  auto initializedModel = (Model *)model;
  if (initializedModel == nullptr) {
    return;
  }
  delete initializedModel;
}