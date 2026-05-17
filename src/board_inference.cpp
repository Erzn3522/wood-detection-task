#include "board_inference.hpp"

#include "io.hpp"

#include <algorithm>

// ---------------------------------------------------------------------------
// IoU + greedy NMS (operates on board-coord detections)
// ---------------------------------------------------------------------------
// Standard IoU between two board-coordinate bboxes.
// Used exclusively for NMS suppression, not for evaluation matching.
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

// Greedy NMS over board-coordinate detections.
// Processes detections in descending confidence order; suppresses any lower-
// confidence box whose IoU with an already-kept box exceeds iou_thresh.
// Applied after stitching all frames to remove cross-frame duplicate detections.
static std::vector<BoardDetection> nms(std::vector<BoardDetection> dets, float iou_thresh)
{
    // Sort by confidence descending so higher-confidence boxes are kept first.
    std::sort(dets.begin(), dets.end(),
              [](const BoardDetection &a, const BoardDetection &b) { return a.conf > b.conf; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<BoardDetection> kept;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i])
            continue;
        kept.push_back(dets[i]);
        // Suppress all remaining boxes that overlap significantly with dets[i].
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > iou_thresh)
                suppressed[j] = true;
        }
    }
    return kept;
}

// ---------------------------------------------------------------------------
// Board prediction
// ---------------------------------------------------------------------------
// Returns the indices of all frames that a board-coordinate bbox overlaps.
// A bbox overlaps frame i if any part of it falls within [offset_i, offset_i + width_i).
// Used to populate the spans_frames field in the output JSON.
static std::vector<int> spans_frames(float x1, float x2, const std::vector<int> &frame_widths)
{
    std::vector<int> result;
    int offset = 0;
    for (int i = 0; i < static_cast<int>(frame_widths.size()); ++i) {
        // Bbox overlaps this frame if it starts before the frame ends and ends after it starts.
        if (x1 < offset + frame_widths[i] && x2 > offset)
            result.push_back(i);
        offset += frame_widths[i];
    }
    return result;
}

namespace board_inference {

// Runs the full per-board inference pipeline:
//   1. Loads each frame PNG and runs YOLO inference in frame-local coordinates.
//   2. Shifts detections to board coordinates by adding the cumulative x offset.
//   3. Applies board-level greedy NMS at IoU=0.5 to remove cross-frame duplicates.
//   4. Computes spans_frames for each surviving detection.
// Board height is taken from the first frame; all frames in a board are the same height.
BoardResult predict_board(const std::vector<FramePath> &frames, const YoloSession &session,
                          float conf_threshold)
{
    BoardResult result;
    result.board_index = frames[0].board_idx;
    result.board_height_px = 0;

    std::vector<BoardDetection> all_dets;
    int x_offset = 0;

    for (const auto &fp : frames) {
        const cv::Mat img = io::load_png(fp.path);
        const int w = img.cols;
        const int h = img.rows;

        result.frames.push_back(fp.path.filename().string());
        result.frame_widths.push_back(w);
        if (result.board_height_px == 0)
            result.board_height_px = h;

        for (const auto &d : session.predict(img, conf_threshold)) {
            // Shift frame-local x coords to board coords using cumulative offset.
            BoardDetection bd;
            bd.x1 = d.x1 + x_offset;
            bd.y1 = d.y1;
            bd.x2 = d.x2 + x_offset;
            bd.y2 = d.y2;
            bd.conf = d.conf;
            all_dets.push_back(bd);
        }

        x_offset += w;
    }

    // Total board width equals the sum of all frame widths.
    result.board_width_px = x_offset;

    // Board-level NMS removes duplicate detections that span frame boundaries.
    auto kept = nms(std::move(all_dets), 0.5f);

    for (auto &bd : kept) {
        bd.spans_frames = spans_frames(bd.x1, bd.x2, result.frame_widths);
        result.knots.push_back(std::move(bd));
    }

    return result;
}

} // namespace board_inference
