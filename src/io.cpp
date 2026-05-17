#include "io.hpp"

#include "board_inference.hpp"
#include "evaluator.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>

namespace io {

// Loads a PNG from disk as a BGR color image.
// Throws if the file does not exist or cannot be decoded.
cv::Mat load_png(const std::filesystem::path &path)
{
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path.string());
    }
    return img;
}

// Serializes a BoardResult to a JSON file under out_dir/{board_index}.json.
// Each knot is written as a 4-corner polygon (axis-aligned bbox corners in
// board coordinates) plus confidence and the list of frame indices it spans.
void write_board_json(const BoardResult &r, const std::filesystem::path &out_dir)
{
    std::filesystem::create_directories(out_dir);

    nlohmann::json j;
    j["board_index"] = r.board_index;
    j["frames"] = r.frames;
    j["frame_widths"] = r.frame_widths;
    j["board_width_px"] = r.board_width_px;
    j["board_height_px"] = r.board_height_px;

    // Encode each knot bbox as a clockwise 4-point polygon: TL, TR, BR, BL.
    auto knots = nlohmann::json::array();
    for (const auto &k : r.knots) {
        nlohmann::json entry;
        entry["polygon"] = {{k.x1, k.y1}, {k.x2, k.y1}, {k.x2, k.y2}, {k.x1, k.y2}};
        entry["confidence"] = k.conf;
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

// Stitches all frames of a board horizontally and draws predicted knot bboxes
// in green with a confidence label. Writes the result to out_dir/images/{board}.png.
// Useful as a quick sanity check without running the full visualizer.
void write_board_image(const BoardResult &r, const std::filesystem::path &frames_dir,
                       const std::filesystem::path &out_dir)
{
    // Load and concatenate all frames left-to-right to form the full board image.
    std::vector<cv::Mat> frames;
    frames.reserve(r.frames.size());
    for (const auto &fname : r.frames)
        frames.push_back(load_png(frames_dir / fname));

    cv::Mat board;
    cv::hconcat(frames, board);

    // Overlay each knot bbox in green with its confidence percentage.
    for (const auto &k : r.knots) {
        const cv::Point pt1(static_cast<int>(k.x1), static_cast<int>(k.y1));
        const cv::Point pt2(static_cast<int>(k.x2), static_cast<int>(k.y2));
        cv::rectangle(board, pt1, pt2, cv::Scalar(0, 255, 0), 2);

        const std::string label = std::to_string(static_cast<int>(k.conf * 100)) + "%";
        cv::putText(board, label, cv::Point(pt1.x, std::max(pt1.y - 4, 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    }

    const auto img_dir = out_dir / "images";
    std::filesystem::create_directories(img_dir);
    const auto out_path = img_dir / (std::to_string(r.board_index) + ".png");
    if (!cv::imwrite(out_path.string(), board))
        throw std::runtime_error("Cannot write image: " + out_path.string());
}

// Writes aggregate evaluation metrics to out_dir/metrics.json.
// Includes IoU threshold, board/frame counts, TP/FP/FN totals,
// precision, recall, F1, and mAP50.
void write_metrics_json(const EvalResult &r, const std::filesystem::path &out_dir)
{
    std::filesystem::create_directories(out_dir);

    nlohmann::json j;
    j["iou_threshold"] = r.iou_threshold;
    j["num_boards"] = r.num_boards;
    j["num_frames"] = r.num_frames;
    j["totals"] = {{"tp", r.totals.tp}, {"fp", r.totals.fp}, {"fn", r.totals.fn}};
    j["precision"] = r.precision;
    j["recall"] = r.recall;
    j["f1"] = r.f1;
    j["map50"] = r.map50;

    const auto out_path = out_dir / "metrics.json";
    std::ofstream f(out_path);
    if (!f)
        throw std::runtime_error("Cannot write: " + out_path.string());
    f << j.dump(2) << '\n';
}

// Writes a human-readable Markdown report to out_dir/REPORT.md.
// Includes an aggregate summary table followed by a per-board breakdown
// sorted by F1 ascending so the worst-performing boards appear first.
void write_report_md(const EvalResult &r, const std::filesystem::path &out_dir)
{
    std::filesystem::create_directories(out_dir);

    // Helper to format floats to 3 decimal places for the Markdown table.
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

    // Sort per-board results by F1 ascending so worst boards appear first.
    auto sorted = r.per_board;
    std::sort(sorted.begin(), sorted.end(),
              [](const BoardMetrics &a, const BoardMetrics &b) { return a.f1 < b.f1; });

    f << "## Per-board results (worst → best F1)\n\n";
    f << "| Board | Frames | TP | FP | FN | Precision | Recall | F1 |\n";
    f << "|---|---|---|---|---|---|---|---|\n";
    for (const auto &m : sorted) {
        f << "| " << m.board_index << " | " << m.num_frames << " | " << m.tp << " | " << m.fp
          << " | " << m.fn << " | " << fmt(m.precision) << " | " << fmt(m.recall) << " | "
          << fmt(m.f1) << " |\n";
    }
    f << '\n';
}

} // namespace io
