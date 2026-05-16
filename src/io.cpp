#include "io.hpp"

#include "board_inference.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
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

void write_board_json(const BoardResult& r, const std::filesystem::path& out_dir)
{
    std::filesystem::create_directories(out_dir);

    nlohmann::json j;
    j["board_index"]    = r.board_index;
    j["frames"]         = r.frames;
    j["frame_widths"]   = r.frame_widths;
    j["board_width_px"] = r.board_width_px;
    j["board_height_px"] = r.board_height_px;

    auto knots = nlohmann::json::array();
    for (const auto& k : r.knots) {
        nlohmann::json entry;
        entry["polygon"] = {{k.x1, k.y1}, {k.x2, k.y1}, {k.x2, k.y2}, {k.x1, k.y2}};
        entry["confidence"]   = k.conf;
        entry["spans_frames"] = k.spans_frames;
        knots.push_back(std::move(entry));
    }
    j["knots"] = std::move(knots);

    const auto out_path = out_dir / (std::to_string(r.board_index) + ".json");
    std::ofstream f(out_path);
    if (!f)
        throw std::runtime_error("Cannot write: " + out_path.string());
    f << j.dump(2) << '\n';
}

} // namespace io
