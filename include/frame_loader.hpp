#pragma once

#include <filesystem>
#include <map>
#include <vector>

struct FramePath {
    int board_idx;
    int frame_idx;
    std::filesystem::path path;
};

namespace frame_loader {

// Scans frames_dir for files matching {boardIdx}_{frameIdx}.png and groups
// them by board index. Frames within each board are sorted by frame index.
std::map<int, std::vector<FramePath>> build_boards(const std::filesystem::path &frames_dir);

} // namespace frame_loader
