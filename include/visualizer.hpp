#pragma once

#include <filesystem>
#include <string>

namespace visualizer
{

enum class Mode {
    Tiled,    // one PNG per board; N frames per row (--cols)
    PerFrame, // one PNG per input frame; header strip + annotations
};

// Reads all board JSONs from predictions_dir, draws predicted knot bboxes
// and (optionally) GT comparison boxes, writes results to out_dir.
//
// Mode::Tiled    — filename: {board_index}.png
// Mode::PerFrame — filename: {board}_{frame}.png (mirrors input frame name)
//
// Color scheme when labels_dir is provided:
//   green  = TP prediction (matched GT at IoU >= 0.5)
//   yellow = FP prediction (unmatched)
//   red    = FN GT box (unmatched)
//   cyan   = matched GT outline (dashed, per-frame mode only)
//   dashed edge + arrow = bbox crossing a frame boundary (per-frame mode only)
// Without labels_dir: all predictions drawn in green.
//
// cols is only used by Mode::Tiled; ignored in Mode::PerFrame.
void run(Mode mode, const std::filesystem::path& predictions_dir,
         const std::filesystem::path& frames_dir, const std::filesystem::path& out_dir,
         const std::filesystem::path& labels_dir, // empty = no GT comparison
         int cols);

} // namespace visualizer
