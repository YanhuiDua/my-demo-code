#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "all_ops.h" // opp/op_proto/built-in/inc

#include "model_build.h"

bool OMModelBuild::GenerateData(ge::Tensor &weight, uint32_t len) {
    char* pdata = new(std::nothrow) char[len];
    if (pdata == nullptr) {
        std::cout << "Invalid Param.len:" << len << std::endl;
        return false;
    }
    for (uint32_t i = 0; i < len; i++) {
        pdata[i] = 1.0;
    }
    auto status = weight.SetData(reinterpret_cast<uint8_t*>(pdata), len);
    if (status != ge::GRAPH_SUCCESS) {
        std::cout << "Set Tensor Data Failed"<< std::endl;
        delete [] pdata;
        return false;
    }
    return true;
}

bool OMModelBuild::GenGraph(ge::Graph& graph) {
     // // input data op => feed
    ge::TensorDesc input_desc(ge::Shape({ 1, 1, 4, 4 }), ge::FORMAT_ND, ge::DT_FLOAT);
    auto input_x = ge::op::Data("input_x");
    input_x.update_input_desc_x(input_desc);
    input_x.update_output_desc_y(input_desc);

    // weight tensor
    auto weight_shape = ge::Shape({ 2, 2, 1, 1 });
    ge::TensorDesc desc_weight_1(weight_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor weight_tensor(desc_weight_1);
    uint32_t weight_1_len = weight_shape.GetShapeSize() * sizeof(float);
    float weight_value = 0.006448820233345032;
    auto status = weight_tensor.SetData(reinterpret_cast<uint8_t*>(&weight_value), weight_1_len);
    if (status != ge::GRAPH_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << "Set Tensor Data Failed" << std::endl;
        return false;
    }

    // const op
    auto conv_weight = ge::op::Const("Conv2D/weight").set_attr_value(weight_tensor);

    // conv op
    auto conv_op = ge::op::Conv2D("conv1");
    conv_op.set_input_x(input_x);
    conv_op.set_input_filter(conv_weight);
    conv_op.set_attr_strides({ 1, 1, 1, 1 });
    conv_op.set_attr_pads({ 0, 0, 0, 0 });
    conv_op.set_attr_dilations({ 1, 1, 1, 1 });

    ge::TensorDesc conv2d_input_desc_x(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc conv2d_input_desc_filter(ge::Shape(), ge::FORMAT_HWCN, ge::DT_FLOAT);
    ge::TensorDesc conv2d_output_desc_y(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    conv_op.update_input_desc_x(conv2d_input_desc_x);
    conv_op.update_input_desc_filter(conv2d_input_desc_filter);
    conv_op.update_output_desc_y(conv2d_output_desc_y);

    auto relu1 = ge::op::Relu("relu");
    relu1.set_input_x(conv_op, "y");

    // Build Graph
    std::vector<ge::Operator> inputs{ input_x };
    std::vector<ge::Operator> outputs{ relu1 };
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{relu1, "y"}};

    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool OMModelBuild::SaveModel(ge::Graph& om_graph, std::string model_path)
{
    LOG(INFO) << "-------Enter: [model_build](SaveModel)-------";
    // 1. Genetate graph
    // ge::Graph om_graph("bias_add_graph");
    // if(!GenGraph(om_graph)) {
    //   LOG(ERROR) << "Generate BiasAdd Graph Failed!");
    // }
    // LOG(INFO) << "Generate BiasAdd Graph SUCCESS!");

    // 2. system init
    std::map<std::string, std::string> global_options = {
        {ge::ir_option::SOC_VERSION, "Ascend310"},
    };
    if (ge::aclgrphBuildInitialize(global_options) !=  ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[model_build](SaveModel) aclgrphBuildInitialize Failed!";
    } else {
      LOG(INFO) << "[model_build](SaveModel) aclgrphBuildInitialize succees";
    }

    // 3. Build IR Model
    ge::ModelBufferData model_om_buffer;
    std::map<std::string, std::string> options;
    //PrepareOptions(options);

    if (ge::aclgrphBuildModel(om_graph, options, model_om_buffer) !=  ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[model_build](SaveModel) aclgrphBuildModel Failed!";
    } else {
        LOG(INFO) << "[model_build](SaveModel) aclgrphBuildModel succees";
    }

    // 4. Save IR Model
    if (ge::aclgrphSaveModel(model_path, model_om_buffer) != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[model_build](SaveModel) aclgrphSaveModel Failed!";
    } else {
        LOG(INFO) << "[model_build](SaveModel) aclgrphSaveModel succees";
    }

    // 5. release resource
    ge::aclgrphBuildFinalize();
    LOG(INFO) << "-------Leave: [model_build](SaveModel)-------";
    return true;
}