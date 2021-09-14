#include <iostream>
#include <memory>
#include <string>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
// torch
#include <torch/torch.h>
#include <torch/script.h>
// user
#include "utils.h"

#define DEFAULT_HEIGHT  1280
#define DEFAULT_WIDTH   720

int main()
{
    auto net = load_model("./traced_net.pt");

    cv::Mat img = cv::imread("img/000004.jpg");
    auto out = frame_prediction(img, net);

    std::cout << out << std::endl;

    // while(1) {
    //     int keycode = cv::waitKey(10) & 0xff;
    //     if (keycode == 27)
    //         break;
    // }
    // // Clean up
    // cv::destroyAllWindows();
    return 0;
}

/*
int main()
{
    // OpenCV camera
    int capture_width = 1280;
    int capture_height = 720;
    int display_width = DEFAULT_WIDTH;
    int display_height = DEFAULT_HEIGHT;
    int framerate = 60;
    int flip_method = 0;

    std::string pipeline = gstreamer_pipeline(capture_width,
                                              capture_height,
                                              display_width,
                                              display_height,
                                              framerate,
                                              flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << std::endl;

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Cannot initialize camera capture" << std::endl;
        return 1;
    }

    std::string window_name = "CSI Camera";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    // cv::Mat img;
    cv::Mat frame;

    std::cout << "Hit esc to exit" << std::endl;

    // Set torch module
    auto net = load_model("../traced_net.pt");

    // while (true)
    // {
    //     if (!cap.read(img))
    //     {
    //         std::cerr << "ERROR: Capture read error" << std::endl;
    //         break;
    //     }

    //     cv::imshow("CSI Camera", img);
    //     int keycode = cv::waitKey(30) & 0xff;
    //     if (keycode == 27)
    //         break;
    // }

    // loop
    while(true)
    {
        cap.read(frame);
        if(frame.empty())
        {
            std::cerr << "ERROR: EMPTY FRAME" << std::endl;
        }

        frame = frame_prediction(frame, net);
        cv::imshow(window_name, frame);

        int keycode = cv::waitKey(30) & 0xff;
        if (keycode == 27)
            break;
    }

    // Clean up
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
*/
