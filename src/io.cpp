#include "io.hpp"

#include "board_inference.hpp"
#include "evaluator.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
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

void write_metrics_json(const EvalResult& r, const std::filesystem::path& out_dir)
{
    std::filesystem::create_directories(out_dir);

    nlohmann::json j;
    j["iou_threshold"] = r.iou_threshold;
    j["num_boards"]    = r.num_boards;
    j["num_frames"]    = r.num_frames;
    j["totals"]        = {{"tp", r.totals.tp}, {"fp", r.totals.fp}, {"fn", r.totals.fn}};
    j["precision"]     = r.precision;
    j["recall"]        = r.recall;
    j["f1"]            = r.f1;
    j["map50"]         = r.map50;

    const auto out_path = out_dir / "metrics.json";
    std::ofstream f(out_path);
    if (!f)
        throw std::runtime_error("Cannot write: " + out_path.string());
    f << j.dump(2) << '\n';
}

void write_report_md(const EvalResult& r, const std::filesystem::path& out_dir)
{
    std::filesystem::create_directories(out_dir);

    auto fmt = [](float v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << v;
        return ss.str();
    };

    std::ofstream f(out_dir / "REPORT.md");
    if (!f)
        throw std::runtime_error("Cannot write REPORT.md");

    f << "# Wood Knot Detection — Evaluation Report\n\n";
    f << "| Metric | Value |\n|---|---|\n";
    f << "| IoU threshold | " << fmt(r.iou_threshold) << " |\n";
    f << "| Boards | " << r.num_boards << " |\n";
    f << "| Frames | " << r.num_frames << " |\n";
    f << "| TP | " << r.totals.tp << " |\n";
    f << "| FP | " << r.totals.fp << " |\n";
    f << "| FN | " << r.totals.fn << " |\n";
    f << "| Precision | " << fmt(r.precision) << " |\n";
    f << "| Recall | " << fmt(r.recall) << " |\n";
    f << "| F1 | " << fmt(r.f1) << " |\n";
    f << "| mAP50 | " << fmt(r.map50) << " |\n\n";

    // Per-board table sorted by F1 ascending (worst first)
    auto sorted = r.per_board;
    std::sort(sorted.begin(), sorted.end(),
              [](const BoardMetrics& a, const BoardMetrics& b) { return a.f1 < b.f1; });

    f << "## Per-board results (worst → best F1)\n\n";
    f << "| Board | Frames | TP | FP | FN | Precision | Recall | F1 |\n";
    f << "|---|---|---|---|---|---|---|---|\n";
    for (const auto& m : sorted) {
        f << "| " << m.board_index << " | " << m.num_frames << " | " << m.tp << " | " << m.fp
          << " | " << m.fn << " | " << fmt(m.precision) << " | " << fmt(m.recall) << " | "
          << fmt(m.f1) << " |\n";
    }
    f << '\n';
}

} // namespace io
