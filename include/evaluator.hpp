#pragma once

#include "board_inference.hpp"

#include <filesystem>
#include <utility>
#include <vector>

struct EvalDetection {
    float conf;
    bool is_tp;
};

struct BoardMetrics {
    int board_index;
    int tp, fp, fn;
    int num_frames;
    float precision, recall, f1;
};

struct EvalResult {
    float iou_threshold;
    int num_boards;
    int num_frames;
    struct {
        int tp, fp, fn;
    } totals;
    float precision, recall, f1, map50;
    std::vector<BoardMetrics> per_board;
};

namespace evaluator {

// Parse YOLO labels for a board and convert to board-coordinate bboxes.
// Uses frame dimensions from the already-computed BoardResult.
std::vector<BoardDetection> parse_gt(const BoardResult &board,
                                     const std::filesystem::path &labels_dir);

// Match predictions against GT at fixed IoU threshold.
// Returns per-board metrics and per-detection TP/FP labels for mAP.
std::pair<BoardMetrics, std::vector<EvalDetection>> match(const BoardResult &pred,
                                                          const std::vector<BoardDetection> &gt,
                                                          float iou_threshold,
                                                          float conf_threshold);

// Compute AP at IoU=0.5 from the collected detection labels.
float compute_map50(std::vector<EvalDetection> all_dets, int num_gt_total);

EvalResult aggregate(std::vector<BoardMetrics> per_board, std::vector<EvalDetection> all_dets,
                     int num_gt_total, int num_frames, float iou_threshold);

} // namespace evaluator
