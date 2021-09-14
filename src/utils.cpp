#include "utils.h"

static int IMG_SIZE = 352;

torch::jit::Module load_model(std::string model_name)
{
    std::string model_dir = model_name;
    auto module = torch::jit::load(model_dir);
    module.to(torch::kCUDA);
    module.eval();
    std::cout << "MODEL LOADED" << std::endl;
    return module;
}
 
torch::Tensor frame_prediction(cv::Mat frame, torch::jit::Module model)
{
    // resize and convert to float
    cv::Mat res_img;
    cv::resize(frame, res_img, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
    cv::Mat flt_img;
    res_img.convertTo(flt_img, CV_32FC3, 1.0f / 255.0f, 0);

    // cv2 to tensor
    torch::Tensor img = 
        torch::from_blob(flt_img.data, {IMG_SIZE, IMG_SIZE, 3});
    img = img.to(torch::kCUDA);
    std::vector<torch::jit::IValue> input;
    input.push_back(img);

    // forward pass
    auto output = model.forward(input).toTensor().detach().to(torch::kCPU);
    return output;
}

std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

std::map<std::string, std::string> load_datafile(std::string data_path)
{
    auto keys = {
        "model_name",
        "epochs",
        "steps",           
        "batch_size",
        "subdivisions",
        "learning_rate",
        "pre_weights",        
        "classes",
        "width",
        "height",           
        "anchor_num",
        "anchors",
        "val",           
        "train",
        "names"
    };
    std::map<std::string, std::string> cfg;
    for (auto &key: keys)
    {
        cfg[key] = "";
    }
    
}
