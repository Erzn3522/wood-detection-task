#pragma once

#include "frame_loader.hpp"
#include "yolo_session.hpp"

#include <string>
#include <vector>

struct BoardDetection
{
    float            x1, y1, x2, y2; // board-coordinate pixels
    float            conf;
    std::vector<int> spans_frames; // frame indices this detection overlaps
};

struct BoardResult
{
    int                       board_index;
    std::vector<std::string>  frames;
    std::vector<int>          frame_widths;
    int                       board_width_px;
    int                       board_height_px;
    std::vector<BoardDetection> knots;
};

namespace board_inference
{

BoardResult predict_board(const std::vector<FramePath>& frames, const YoloSession& session,
                          float conf_threshold);

} // namespace board_inference
