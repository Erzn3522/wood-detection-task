#include "visualizer.hpp"

#include "board_inference.hpp"
#include "evaluator.hpp"
#include "io.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

static constexpr int kTileW = 640;  // tiled mode: tile column width
static constexpr int kPadding = 24; // tiled mode: pixels below each row for frame label
static constexpr int kHeaderH = 40; // per-frame mode: header strip height

// ---------------------------------------------------------------------------
// Parse a board JSON back into a BoardResult (shared by both modes)
// ---------------------------------------------------------------------------
// Deserializes a board JSON file written by io::write_board_json back into a BoardResult.
// Polygon corners are stored as [TL, TR, BR, BL]; x1/y1 come from index 0, x2/y2 from index 2.
static BoardResult parse_board_json(const std::filesystem::path &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open: " + path.string());

    const auto j = nlohmann::json::parse(f);

    BoardResult r;
    r.board_index = j["board_index"];
    r.frames = j["frames"].get<std::vector<std::string>>();
    r.frame_widths = j["frame_widths"].get<std::vector<int>>();
    r.board_width_px = j["board_width_px"];
    r.board_height_px = j["board_height_px"];

    for (const auto &k : j["knots"]) {
        BoardDetection bd;
        // Reconstruct bbox from TL (index 0) and BR (index 2) polygon corners.
        bd.x1 = k["polygon"][0][0];
        bd.y1 = k["polygon"][0][1];
        bd.x2 = k["polygon"][2][0];
        bd.y2 = k["polygon"][2][1];
        bd.conf = k["confidence"];
        bd.spans_frames = k["spans_frames"].get<std::vector<int>>();
        r.knots.push_back(bd);
    }
    return r;
}

// ---------------------------------------------------------------------------
// IoU and IoMin for matching (board-coord space)
// ---------------------------------------------------------------------------
// Standard IoU for visualization matching; mirrors evaluator.cpp logic.
// Returns inter / (area_a + area_b - inter).
static float iou(const BoardDetection &a, const BoardDetection &b)
{
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    if (iw == 0.0f || ih == 0.0f)
        return 0.0f;
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

// Intersection over minimum area; catches containment cases where IoU is low.
// Returns inter / min(area_a, area_b).
static float iomin(const BoardDetection &a, const BoardDetection &b)
{
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    if (iw == 0.0f || ih == 0.0f)
        return 0.0f;
    const float inter = iw * ih;
    const float min_area = std::min((a.x2 - a.x1) * (a.y2 - a.y1), (b.x2 - b.x1) * (b.y2 - b.y1));
    return min_area > 0.0f ? inter / min_area : 0.0f;
}

// ---------------------------------------------------------------------------
// Match predictions → tp_flags; marks gt_matched for matched GT boxes
// ---------------------------------------------------------------------------
// Greedy TP/FP assignment for visualization coloring.
// Mirrors the evaluator matching logic: predictions are processed in descending
// confidence order, each GT is matched at most once, threshold is 0.5.
// Fills gt_matched in place so the caller can identify unmatched GT (FN) boxes.
static std::vector<bool> match_predictions(const std::vector<BoardDetection> &knots,
                                           const std::vector<BoardDetection> &gt,
                                           std::vector<bool> &gt_matched)
{
    std::vector<bool> tp(knots.size(), false);
    std::vector<size_t> order(knots.size());
    std::iota(order.begin(), order.end(), 0);
    // Process highest-confidence predictions first.
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return knots[a].conf > knots[b].conf; });

    for (size_t idx : order) {
        float best = 0.0f;
        int best_gi = -1;
        for (size_t gi = 0; gi < gt.size(); ++gi) {
            if (gt_matched[gi])
                continue;
            const float v = std::max(iou(knots[idx], gt[gi]), iomin(knots[idx], gt[gi]));
            if (v > best) {
                best = v;
                best_gi = static_cast<int>(gi);
            }
        }
        if (best_gi >= 0 && best >= 0.5f) {
            tp[idx] = true;
            gt_matched[best_gi] = true;
        }
    }
    return tp;
}

// ===========================================================================
// Tiled mode helpers
// ===========================================================================

// Draws a board-coordinate bbox onto a tiled canvas, clipped to the tile(s) it overlaps.
// A bbox spanning multiple frames is drawn once per overlapping tile, clipped at each tile edge.
static void draw_box_in_tile(cv::Mat &canvas, float board_x1, float board_y1, float board_x2,
                             float board_y2, const std::vector<int> &frame_widths, int row_height,
                             int cols, const cv::Scalar &color, const std::string &label = "")
{
    int x_off = 0;
    for (int fi = 0; fi < static_cast<int>(frame_widths.size()); ++fi) {
        const int fw = frame_widths[fi];
        if (board_x2 > x_off && board_x1 < x_off + fw) {
            // Map frame fi to its tile position in the grid.
            const int row = fi / cols;
            const int col = fi % cols;
            const int tile_x = col * kTileW;
            const int tile_y = row * row_height;

            // Clip bbox to the tile's x range, then offset to canvas coordinates.
            const int bx1 = std::max(0, static_cast<int>(board_x1 - x_off)) + tile_x;
            const int by1 = static_cast<int>(board_y1) + tile_y;
            const int bx2 = std::min(fw, static_cast<int>(board_x2 - x_off)) + tile_x;
            const int by2 = static_cast<int>(board_y2) + tile_y;

            cv::rectangle(canvas, cv::Point(bx1, by1), cv::Point(bx2, by2), color, 2);
            if (!label.empty()) {
                const int text_y = std::max(tile_y + 10, by1 - 3);
                cv::putText(canvas, label, cv::Point(bx1, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                            color, 1, cv::LINE_AA);
            }
        }
        x_off += fw;
    }
}

// Tiled mode: lays out all frames of each board in a grid (N rows x cols columns).
// Each cell is kTileW wide with a kPadding strip below for the frame index label.
// Prediction bboxes are drawn in green (no GT) or TP/FP colors (with GT).
// FN GT boxes are drawn in red. Output: one PNG per board.
static void run_tiled_impl(const std::filesystem::path &predictions_dir,
                           const std::filesystem::path &frames_dir,
                           const std::filesystem::path &out_dir,
                           const std::filesystem::path &labels_dir, int cols)
{
    if (!std::filesystem::is_directory(predictions_dir)) {
        std::cerr << "[error] predictions directory not found: " << predictions_dir
                  << "\nRun 'predict' or 'test' first to generate predictions.\n";
        return;
    }
    std::filesystem::create_directories(out_dir);
    const bool has_gt = !labels_dir.empty() && std::filesystem::is_directory(labels_dir);

    std::vector<std::filesystem::path> json_files;
    for (const auto &entry : std::filesystem::directory_iterator(predictions_dir)) {
        if (entry.path().extension() == ".json")
            json_files.push_back(entry.path());
    }
    std::sort(json_files.begin(), json_files.end());

    for (const auto &json_path : json_files) {
        const BoardResult result = parse_board_json(json_path);
        const int n = static_cast<int>(result.frames.size());
        const int fh = result.board_height_px;
        const int rows = (n + cols - 1) / cols;
        const int row_h = fh + kPadding;

        cv::Mat canvas(rows * row_h, cols * kTileW, CV_8UC3, cv::Scalar(30, 30, 30));

        for (int fi = 0; fi < n; ++fi) {
            const cv::Mat frame = io::load_png(frames_dir / result.frames[fi]);
            const int row = fi / cols;
            const int col = fi % cols;
            const int tx = col * kTileW;
            const int ty = row * row_h;

            frame.copyTo(canvas(cv::Rect(tx, ty, frame.cols, frame.rows)));

            cv::putText(canvas,
                        "#" + std::to_string(result.frames[fi].rfind('_') != std::string::npos
                                                 ? std::stoi(result.frames[fi].substr(
                                                       result.frames[fi].rfind('_') + 1,
                                                       result.frames[fi].size() - 4 -
                                                           result.frames[fi].rfind('_') - 1))
                                                 : fi),
                        cv::Point(tx + 3, ty + fh + 16), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                        cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
        }

        std::vector<bool> gt_matched;
        std::vector<bool> tp_flags;
        std::vector<BoardDetection> gt;

        if (has_gt) {
            gt = evaluator::parse_gt(result, labels_dir);
            gt_matched = std::vector<bool>(gt.size(), false);
            tp_flags = match_predictions(result.knots, gt, gt_matched);
        }

        for (size_t i = 0; i < result.knots.size(); ++i) {
            const auto &k = result.knots[i];
            cv::Scalar color = has_gt
                                   ? (tp_flags[i] ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 220, 255))
                                   : cv::Scalar(0, 220, 0);
            const std::string lbl = std::to_string(static_cast<int>(k.conf * 100)) + "%";
            draw_box_in_tile(canvas, k.x1, k.y1, k.x2, k.y2, result.frame_widths, row_h, cols,
                             color, lbl);
        }

        if (has_gt) {
            for (size_t gi = 0; gi < gt.size(); ++gi) {
                if (gt_matched[gi])
                    continue;
                draw_box_in_tile(canvas, gt[gi].x1, gt[gi].y1, gt[gi].x2, gt[gi].y2,
                                 result.frame_widths, row_h, cols, cv::Scalar(0, 0, 220), "FN");
            }
        }

        const auto out_path = out_dir / (std::to_string(result.board_index) + ".png");
        if (!cv::imwrite(out_path.string(), canvas))
            throw std::runtime_error("Cannot write: " + out_path.string());

        std::cout << "board " << result.board_index << " -> " << out_path.filename().string()
                  << '\n';
    }
}

// ===========================================================================
// Per-frame mode helpers
// ===========================================================================

struct FrameStats {
    int tp = 0, fp = 0, fn = 0;
    int tp_cross = 0, fp_cross = 0; // subset of tp/fp that span a frame boundary
};

// Returns the cumulative x offset (in board pixels) at the start of frame fi.
static int frame_x_offset(int fi, const std::vector<int> &fw)
{
    int off = 0;
    for (int i = 0; i < fi; ++i)
        off += fw[i];
    return off;
}

// Returns which edges of the bbox bleed outside frame fi:
//   -1 = no bleed  0 = left edge  1 = right edge  2 = both edges
static int bleeding_edge(const BoardDetection &d, int fi, const std::vector<int> &fw)
{
    const int x_off = frame_x_offset(fi, fw);
    const int x_end = x_off + fw[fi];
    const bool left = static_cast<int>(d.x1) < x_off;
    const bool right = static_cast<int>(d.x2) > x_end;
    if (left && right)
        return 2;
    if (left)
        return 0;
    if (right)
        return 1;
    return -1;
}

// Draws a dashed line between p1 and p2 by alternating filled and skipped segments.
// dash/gap are lengths in pixels; parameterized along the line direction.
static void draw_dashed_line(cv::Mat &img, cv::Point p1, cv::Point p2, const cv::Scalar &color,
                             int thickness, int dash = 7, int gap = 4)
{
    const double dx = p2.x - p1.x;
    const double dy = p2.y - p1.y;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1.0)
        return;
    const double step = dash + gap;
    const int n_segs = static_cast<int>(len / step) + 1;
    for (int i = 0; i < n_segs; ++i) {
        // t1/t2 are fractional positions along the line [0, 1].
        const double t1 = std::min(1.0, i * step / len);
        const double t2 = std::min(1.0, (i * step + dash) / len);
        cv::line(img, cv::Point(static_cast<int>(p1.x + t1 * dx), static_cast<int>(p1.y + t1 * dy)),
                 cv::Point(static_cast<int>(p1.x + t2 * dx), static_cast<int>(p1.y + t2 * dy)),
                 color, thickness, cv::LINE_AA);
    }
}

// Draws a rectangle where bleeding edges are dashed; solid otherwise.
// bleed: -1 = solid all, 0 = left dashed, 1 = right dashed, 2 = both dashed
static void draw_mixed_rect(cv::Mat &img, cv::Point p1, cv::Point p2, const cv::Scalar &color,
                            int thickness, int bleed)
{
    const bool dash_left = (bleed == 0 || bleed == 2);
    const bool dash_right = (bleed == 1 || bleed == 2);

    // Top and bottom are always solid (frames share full height)
    cv::line(img, p1, cv::Point(p2.x, p1.y), color, thickness, cv::LINE_AA);
    cv::line(img, cv::Point(p1.x, p2.y), p2, color, thickness, cv::LINE_AA);

    if (dash_left)
        draw_dashed_line(img, p1, cv::Point(p1.x, p2.y), color, thickness);
    else
        cv::line(img, p1, cv::Point(p1.x, p2.y), color, thickness, cv::LINE_AA);

    if (dash_right)
        draw_dashed_line(img, cv::Point(p2.x, p1.y), p2, color, thickness);
    else
        cv::line(img, cv::Point(p2.x, p1.y), p2, color, thickness, cv::LINE_AA);
}

static void draw_dashed_rect_full(cv::Mat &img, cv::Point p1, cv::Point p2, const cv::Scalar &color,
                                  int thickness)
{
    draw_dashed_line(img, p1, cv::Point(p2.x, p1.y), color, thickness);
    draw_dashed_line(img, cv::Point(p1.x, p2.y), p2, color, thickness);
    draw_dashed_line(img, p1, cv::Point(p1.x, p2.y), color, thickness);
    draw_dashed_line(img, cv::Point(p2.x, p1.y), p2, color, thickness);
}

// Small filled triangle arrow centered at (cx, cy).
// dir = +1 -> pointing right (->); dir = -1 -> pointing left (<-)
static void draw_arrow_marker(cv::Mat &img, int cx, int cy, int dir, const cv::Scalar &color)
{
    const int r = 5;
    cv::Point pts[3] = {cv::Point(cx + dir * r, cy), cv::Point(cx - dir * (r / 2), cy - r),
                        cv::Point(cx - dir * (r / 2), cy + r)};
    const cv::Point *ppts[1] = {pts};
    const int npts = 3;
    cv::fillPoly(img, ppts, &npts, 1, color);
}

static void render_header(cv::Mat &hdr, const std::string &fname, int fi, int n,
                          const FrameStats &st, bool has_gt)
{
    hdr.setTo(cv::Scalar(40, 40, 40));

    const std::string row1 = fname + "   frame " + std::to_string(fi + 1) + "/" + std::to_string(n);
    cv::putText(hdr, row1, cv::Point(4, 14), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    if (!has_gt) {
        // st.tp repurposed as "pred count" when GT is unavailable
        cv::putText(hdr, "pred:" + std::to_string(st.tp), cv::Point(4, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
        cv::rectangle(hdr, cv::Point(60, 24), cv::Point(69, 33), cv::Scalar(0, 220, 0), cv::FILLED);
        cv::putText(hdr, "pred", cv::Point(71, 33), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                    cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
        return;
    }

    const std::string cross_tp = st.tp_cross > 0 ? " (" + std::to_string(st.tp_cross) + "<>)" : "";
    const std::string cross_fp = st.fp_cross > 0 ? "(" + std::to_string(st.fp_cross) + "<>)" : "";

    const std::string row2 = "TP:" + std::to_string(st.tp) + cross_tp +
                             "  FP:" + std::to_string(st.fp) + cross_fp +
                             "  FN:" + std::to_string(st.fn);
    cv::putText(hdr, row2, cv::Point(4, 32), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    // Color swatches on the right side of the header
    int sx = std::max(hdr.cols - 130, 180);
    const int sy1 = 24, sy2 = 33;

    const auto swatch = [&](const cv::Scalar &c, const std::string &lbl) {
        cv::rectangle(hdr, cv::Point(sx, sy1), cv::Point(sx + 9, sy2), c, cv::FILLED);
        cv::putText(hdr, lbl, cv::Point(sx + 11, sy2), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                    cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
        sx += 28;
    };

    swatch(cv::Scalar(0, 220, 0), "TP");
    swatch(cv::Scalar(0, 220, 255), "FP");
    swatch(cv::Scalar(0, 0, 220), "FN");

    // Matched-GT swatch: short dashed cyan line (not a filled square)
    draw_dashed_line(hdr, cv::Point(sx, (sy1 + sy2) / 2), cv::Point(sx + 9, (sy1 + sy2) / 2),
                     cv::Scalar(220, 220, 0), 1);
    cv::putText(hdr, "GT", cv::Point(sx + 11, sy2), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
}

// Per-frame mode: writes one annotated PNG per input frame.
// Each output has a 40 px dark header with frame name, TP/FP/FN counts and legend,
// followed by the original frame with overlaid prediction and GT boxes.
// Cross-frame knots are marked with dashed edges and arrow markers.
static void run_per_frame_impl(const std::filesystem::path &predictions_dir,
                               const std::filesystem::path &frames_dir,
                               const std::filesystem::path &out_dir,
                               const std::filesystem::path &labels_dir)
{
    if (!std::filesystem::is_directory(predictions_dir)) {
        std::cerr << "[error] predictions directory not found: " << predictions_dir
                  << "\nRun 'predict' or 'test' first to generate predictions.\n";
        return;
    }
    std::filesystem::create_directories(out_dir);
    const bool has_gt = !labels_dir.empty() && std::filesystem::is_directory(labels_dir);

    std::vector<std::filesystem::path> json_files;
    for (const auto &entry : std::filesystem::directory_iterator(predictions_dir)) {
        if (entry.path().extension() == ".json")
            json_files.push_back(entry.path());
    }
    std::sort(json_files.begin(), json_files.end());

    for (const auto &json_path : json_files) {
        const BoardResult result = parse_board_json(json_path);
        const int n = static_cast<int>(result.frames.size());

        std::vector<bool> gt_matched;
        std::vector<bool> tp_flags;
        std::vector<BoardDetection> gt;

        if (has_gt) {
            gt = evaluator::parse_gt(result, labels_dir);
            gt_matched = std::vector<bool>(gt.size(), false);
            tp_flags = match_predictions(result.knots, gt, gt_matched);
        }

        for (int fi = 0; fi < n; ++fi) {
            const cv::Mat frame = io::load_png(frames_dir / result.frames[fi]);
            const int fw = result.frame_widths[fi];
            const int fh = frame.rows;
            const int x_off = frame_x_offset(fi, result.frame_widths);

            // Canvas = header strip on top + frame below
            cv::Mat canvas(kHeaderH + fh, frame.cols, CV_8UC3, cv::Scalar(30, 30, 30));
            frame.copyTo(canvas(cv::Rect(0, kHeaderH, frame.cols, fh)));

            FrameStats stats;

            // --- Draw predictions ---
            for (size_t ki = 0; ki < result.knots.size(); ++ki) {
                const auto &k = result.knots[ki];

                // Intersection test with frame fi's x range in board coords
                if (static_cast<int>(k.x2) <= x_off || static_cast<int>(k.x1) >= x_off + fw)
                    continue;

                // Clip to frame-local coordinates
                const int lx1 = std::max(0, static_cast<int>(k.x1) - x_off);
                const int lx2 = std::min(fw, static_cast<int>(k.x2) - x_off);
                const int ly1 = static_cast<int>(k.y1);
                const int ly2 = static_cast<int>(k.y2);

                // Shift y down by header height for canvas placement
                const cv::Point cp1(lx1, kHeaderH + ly1);
                const cv::Point cp2(lx2, kHeaderH + ly2);

                const int bleed = bleeding_edge(k, fi, result.frame_widths);
                const bool is_cross = (bleed != -1);

                cv::Scalar color;
                if (has_gt) {
                    color = tp_flags[ki] ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 220, 255);
                    if (tp_flags[ki]) {
                        ++stats.tp;
                        if (is_cross)
                            ++stats.tp_cross;
                    } else {
                        ++stats.fp;
                        if (is_cross)
                            ++stats.fp_cross;
                    }
                } else {
                    color = cv::Scalar(0, 220, 0);
                    ++stats.tp; // repurposed as pred count in no-GT mode
                    if (is_cross)
                        ++stats.tp_cross;
                }

                draw_mixed_rect(canvas, cp1, cp2, color, 2, bleed);

                // Arrow markers on bleeding edges so viewer knows the bbox continues
                const int mid_y = kHeaderH + (ly1 + ly2) / 2;
                if (bleed == 0 || bleed == 2) // <- marker: bbox continues into prev frame
                    draw_arrow_marker(canvas, lx1 + 8, mid_y, -1, color);
                if (bleed == 1 || bleed == 2) // -> marker: bbox continues into next frame
                    draw_arrow_marker(canvas, lx2 - 8, mid_y, +1, color);

                // Label: confidence + stable knot ID
                const std::string lbl =
                    std::to_string(static_cast<int>(k.conf * 100)) + "% #K" + std::to_string(ki);
                const int text_y = std::max(kHeaderH + 10, cp1.y - 3);
                cv::putText(canvas, lbl, cv::Point(lx1 + 2, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.35,
                            color, 1, cv::LINE_AA);
            }

            // --- Draw GT boxes ---
            if (has_gt) {
                for (size_t gi = 0; gi < gt.size(); ++gi) {
                    const auto &g = gt[gi];

                    // GT coords are in board space; filter to current frame
                    if (static_cast<int>(g.x2) <= x_off || static_cast<int>(g.x1) >= x_off + fw)
                        continue;

                    const int glx1 = std::max(0, static_cast<int>(g.x1) - x_off);
                    const int glx2 = std::min(fw, static_cast<int>(g.x2) - x_off);
                    const int gly1 = static_cast<int>(g.y1);
                    const int gly2 = static_cast<int>(g.y2);

                    const cv::Point gp1(glx1, kHeaderH + gly1);
                    const cv::Point gp2(glx2, kHeaderH + gly2);

                    if (gt_matched[gi]) {
                        // Cyan dashed outline: visual IoU reference for matched GT
                        draw_dashed_rect_full(canvas, gp1, gp2, cv::Scalar(220, 220, 0), 1);
                    } else {
                        // FN: solid red box
                        cv::rectangle(canvas, gp1, gp2, cv::Scalar(0, 0, 220), 2);
                        cv::putText(canvas, "FN", cv::Point(glx1 + 2, kHeaderH + gly1 + 12),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 220), 1,
                                    cv::LINE_AA);
                        ++stats.fn;
                    }
                }
            }

            // Render header after stats are complete
            cv::Mat hdr_roi = canvas(cv::Rect(0, 0, frame.cols, kHeaderH));
            render_header(hdr_roi, result.frames[fi], fi, n, stats, has_gt);

            // Small footnote when cross-frame knots are present
            if (stats.tp_cross + stats.fp_cross > 0) {
                cv::putText(canvas, "cross-frame knots merged in board.json",
                            cv::Point(4, canvas.rows - 4), cv::FONT_HERSHEY_SIMPLEX, 0.28,
                            cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
            }

            const auto out_path = out_dir / result.frames[fi];
            if (!cv::imwrite(out_path.string(), canvas))
                throw std::runtime_error("Cannot write: " + out_path.string());
        }
        std::cout << "board " << result.board_index << " -> " << n << " frames\n";
    }
}

// ===========================================================================
// visualizer::run — public dispatcher
// ===========================================================================
namespace visualizer {

// Dispatches to the selected visualization mode.
// cols is only used in Tiled mode; ignored in PerFrame mode.
void run(Mode mode, const std::filesystem::path &predictions_dir,
         const std::filesystem::path &frames_dir, const std::filesystem::path &out_dir,
         const std::filesystem::path &labels_dir, int cols)
{
    if (mode == Mode::Tiled)
        run_tiled_impl(predictions_dir, frames_dir, out_dir, labels_dir, cols);
    else
        run_per_frame_impl(predictions_dir, frames_dir, out_dir, labels_dir);
}

} // namespace visualizer
