#pragma once

#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

struct Detection {
    float x1, y1, x2, y2; // pixel coords in original frame space
    float conf;
    int cls;
};

class YoloSession
{
public:
    YoloSession(const std::filesystem::path &model_path, const std::string &device);
    ~YoloSession();

    std::vector<Detection> predict(const cv::Mat &bgr_frame, float conf_threshold) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
