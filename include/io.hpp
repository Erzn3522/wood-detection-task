#pragma once

#include <filesystem>
#include <opencv2/core/mat.hpp>

namespace io
{

// Loads a PNG image from path and returns it as a BGR cv::Mat.
// Throws std::runtime_error if the file cannot be read.
cv::Mat load_png(const std::filesystem::path& path);

} // namespace io
