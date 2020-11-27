#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include <paddle_api.h>

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE = paddle::lite_api::PowerMode::LITE_POWER_HIGH;

const std::string model_path = "../train/saved_infer_model";
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 4, 4};


template <typename T>
std::string data_to_string(const T* data, const int64_t size) {
  std::ostringstream ss;
  ss << "[";
  for (int64_t i = 0; i < size - 1; ++i) {
    ss << std::setprecision(2) << std::setw(9) << std::setfill(' ') 
       << std::fixed << data[i] << ", ";
  }
  ss << std::setprecision(2) << std::setw(9) << std::setfill(' ') 
     << std::fixed << data[size - 1] << "]";
  // ss << data[size - 1] << "]";
  return ss.str();
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream ss;
  if (shape.empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ss << shape[i] << ", ";
  }
  ss << shape[shape.size() - 1] << "}";
  return ss.str();
}

template <typename T>
void tensor_to_string(const T* data, const std::vector<int64_t>& shape) {
  std::cout << "Shape: " << shape_to_string(shape) << std::endl;
  int64_t stride = shape.back(); 
  int64_t index = 0;
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      std::cout << std::endl;
      for (size_t k = 0; k < shape[2]; ++k) {
        const T * data_start = data + index;
        std::cout << data_to_string<T>(data_start, stride) << std::endl;
        index += stride;
      }
    }
  }
}

int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

int64_t shape_production(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  auto* input_data = input_tensor->mutable_data<float>();
  for (int64_t i = 0; i < shape_production(INPUT_SHAPE); ++i) {
    input_data[i] = i + 1.0f;
  }

  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  // 3. Repeat Run
  auto start_time = get_current_us();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }
  auto end_time = get_current_us();
  // 4. Speed Report
  std::cout << "================== Speed Report ===================" << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats 
            << ", spend " << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average." << std::endl;

  // 5. Get all output
  std::cout << std::endl << "Output Index: <0>" << std::endl;
  tensor_to_string<float>(input_data, input_tensor->shape());
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    std::cout << std::endl << "Output Index: <" << i << ">" << std::endl;
    tensor_to_string<float>(output_data, output_tensor->shape());
  }
}

void RunLiteModel(const std::string model_path) {
  // 1. Create MobileConfig
  auto start_time = get_current_us();
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  // std::string model_buffer = ReadFile(model_path+".nb");
  // mobile_config.set_model_from_buffer(model_buffer);
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "MobileConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  auto end_time = get_current_us();

  // 3. Run model
  process(predictor);
  std::cout << "MobileConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;
}

#ifdef USE_FULL_API
void RunFullModel(const std::string model_path) {
  // 1. Create CxxConfig
  auto start_time = get_current_us();
  CxxConfig cxx_config;
  cxx_config.set_model_file(model_path + "_opt/model");
  cxx_config.set_param_file(model_path + "_opt/params");
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                               Place{TARGET(kHost), PRECISION(kFloat)}});
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "CxxConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  auto end_time = get_current_us();
  // 3. Run model
  process(predictor);
  std::cout << "CxxConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;
}

void SaveOptModel(const std::string model_path, const int model_type = 0) {
  // 1. Create CxxConfig
  CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_path + "/__model__");
    cxx_config.set_param_file(model_path + "/__params__");
  } else {
    cxx_config.set_model_dir(model_path);
  }
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                               Place{TARGET(kHost), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "CxxConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Save optimized model
  predictor->SaveOptimizedModel(model_path, LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;

  predictor->SaveOptimizedModel(model_path+"_opt", LiteModelType::kProtobuf);
  std::cout << "Save optimized model to " << (model_path+"_opt") << std::endl;
}
#endif

int main(int argc, char **argv) {

#ifdef USE_FULL_API
  SaveOptModel(model_path);
  RunFullModel(model_path);
#endif

  RunLiteModel(model_path);

  return 0;
}