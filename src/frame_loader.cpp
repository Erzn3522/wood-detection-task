#include "frame_loader.hpp"

#include <algorithm>
#include <regex>
#include <stdexcept>

namespace frame_loader {

// Scans frames_dir for PNG files matching the {boardIdx}_{frameIdx}.png naming
// convention, groups them by board index, and sorts each group by frame index.
// Files that do not match the pattern are silently skipped.
// Returns a map from board index to its ordered list of frame paths.
std::map<int, std::vector<FramePath>> build_boards(const std::filesystem::path &frames_dir)
{
    if (!std::filesystem::is_directory(frames_dir)) {
        throw std::runtime_error("Not a directory: " + frames_dir.string());
    }

    // Matches filenames like "1041_3.png"; captures board index and frame index.
    static const std::regex kPattern{R"(^(\d+)_(\d+)\.png$)"};
    std::map<int, std::vector<FramePath>> boards;

    for (const auto &entry : std::filesystem::directory_iterator(frames_dir)) {
        if (!entry.is_regular_file())
            continue;

        const std::string name = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, kPattern))
            continue;

        // Extract board and frame indices from regex capture groups.
        FramePath fp;
        fp.board_idx = std::stoi(m[1].str());
        fp.frame_idx = std::stoi(m[2].str());
        fp.path = entry.path();
        boards[fp.board_idx].push_back(fp);
    }

    // Sort each board's frames by frame index so stitching order is correct.
    for (auto &[board_idx, frames] : boards) {
        std::sort(frames.begin(), frames.end(),
                  [](const FramePath &a, const FramePath &b) { return a.frame_idx < b.frame_idx; });
    }

    return boards;
}

} // namespace frame_loader
