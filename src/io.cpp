#include "io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace io
{

cv::Mat load_png(const std::filesystem::path& path)
{
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path.string());
    }
    return img;
}

} // namespace io
