#pragma once

#include <filesystem>
#include <opencv2/core/mat.hpp>

struct BoardResult; // defined in board_inference.hpp

namespace io
{

// Loads a PNG image from path and returns it as a BGR cv::Mat.
// Throws std::runtime_error if the file cannot be read.
cv::Mat load_png(const std::filesystem::path& path);

// Serialises a BoardResult to {out_dir}/{board_index}.json.
// Creates out_dir if it does not exist.
void write_board_json(const BoardResult& result, const std::filesystem::path& out_dir);

} // namespace io
