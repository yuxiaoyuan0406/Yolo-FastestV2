#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <map>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
// torch
#include <torch/torch.h>
#include <torch/script.h>

torch::Tensor frame_prediction(cv::Mat frame, torch::jit::Module model);

torch::jit::Module load_model(std::string model_name);

std::string gstreamer_pipeline(
    int capture_width, 
    int capture_height, 
    int display_width, 
    int display_height, 
    int framerate, 
    int flip_method);

std::map<std::string, std::string> load_datafile(std::string data_path);
