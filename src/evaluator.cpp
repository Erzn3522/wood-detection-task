#include "evaluator.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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

static float safe_div(float num, float den) { return den > 0.0f ? num / den : 0.0f; }

// ---------------------------------------------------------------------------
// namespace evaluator
// ---------------------------------------------------------------------------
namespace evaluator {

std::vector<BoardDetection> parse_gt(const BoardResult &board,
                                     const std::filesystem::path &labels_dir)
{
    std::vector<BoardDetection> gt;
    int x_offset = 0;

    for (size_t fi = 0; fi < board.frames.size(); ++fi) {
        const int fw = board.frame_widths[fi];
        const int fh = board.board_height_px;
        const float W = static_cast<float>(fw);
        const float H = static_cast<float>(fh);

        // Derive label filename from image filename (replace .png → .txt)
        std::string label_name = board.frames[fi];
        label_name.replace(label_name.size() - 4, 4, ".txt");
        const auto label_path = labels_dir / label_name;

        std::ifstream f(label_path);
        if (!f)
            continue; // frames without annotations are valid (no knots)

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty())
                continue;
            std::istringstream ss(line);
            int cls;
            float cx, cy, w, h;
            ss >> cls >> cx >> cy >> w >> h;

            const float x1 = (cx - w / 2.0f) * W + x_offset;
            const float y1 = (cy - h / 2.0f) * H;
            const float x2 = (cx + w / 2.0f) * W + x_offset;
            const float y2 = (cy + h / 2.0f) * H;

            BoardDetection bd;
            bd.x1 = x1;
            bd.y1 = y1;
            bd.x2 = x2;
            bd.y2 = y2;
            bd.conf = 1.0f;
            gt.push_back(bd);
        }

        x_offset += fw;
    }

    return gt;
}

std::pair<BoardMetrics, std::vector<EvalDetection>> match(const BoardResult &pred,
                                                          const std::vector<BoardDetection> &gt,
                                                          float iou_threshold, float conf_threshold)
{
    // Sort predictions by confidence descending for greedy matching
    std::vector<size_t> order(pred.knots.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return pred.knots[a].conf > pred.knots[b].conf; });

    std::vector<bool> gt_matched(gt.size(), false);
    std::vector<EvalDetection> eval_dets;

    for (size_t idx : order) {
        const auto &p = pred.knots[idx];
        float best_iou = 0.0f;
        int best_gt = -1;

        for (size_t gi = 0; gi < gt.size(); ++gi) {
            if (gt_matched[gi])
                continue;
            const float v = std::max(iou(p, gt[gi]), iomin(p, gt[gi]));
            if (v > best_iou) {
                best_iou = v;
                best_gt = static_cast<int>(gi);
            }
        }

        const bool is_tp = (best_gt >= 0 && best_iou >= iou_threshold);
        if (is_tp)
            gt_matched[best_gt] = true;

        eval_dets.push_back({p.conf, is_tp});
    }

    // Count TP/FP/FN at the given conf threshold
    int tp = 0, fp = 0;
    for (const auto &d : eval_dets) {
        if (d.conf < conf_threshold)
            continue;
        if (d.is_tp)
            ++tp;
        else
            ++fp;
    }
    const int fn = static_cast<int>(gt.size()) - tp;

    BoardMetrics m;
    m.board_index = pred.board_index;
    m.num_frames = static_cast<int>(pred.frames.size());
    m.tp = tp;
    m.fp = fp;
    m.fn = std::max(0, fn);
    m.precision = safe_div(tp, tp + fp);
    m.recall = safe_div(tp, tp + m.fn);
    m.f1 = safe_div(2.0f * m.precision * m.recall, m.precision + m.recall);

    return {m, eval_dets};
}

float compute_map50(std::vector<EvalDetection> all_dets, int num_gt_total)
{
    if (num_gt_total == 0)
        return 0.0f;

    std::sort(all_dets.begin(), all_dets.end(),
              [](const EvalDetection &a, const EvalDetection &b) { return a.conf > b.conf; });

    int cum_tp = 0, cum_fp = 0;
    std::vector<float> prec, rec;

    for (const auto &d : all_dets) {
        if (d.is_tp)
            ++cum_tp;
        else
            ++cum_fp;
        prec.push_back(safe_div(cum_tp, cum_tp + cum_fp));
        rec.push_back(static_cast<float>(cum_tp) / num_gt_total);
    }

    if (prec.empty())
        return 0.0f;

    // Make precision monotonically non-increasing from right to left
    for (int i = static_cast<int>(prec.size()) - 2; i >= 0; --i)
        prec[i] = std::max(prec[i], prec[i + 1]);

    // Trapezoid integration under the PR curve
    float ap = 0.0f;
    for (size_t i = 1; i < rec.size(); ++i)
        ap += (rec[i] - rec[i - 1]) * prec[i];

    return std::max(0.0f, ap);
}

EvalResult aggregate(std::vector<BoardMetrics> per_board, std::vector<EvalDetection> all_dets,
                     int num_gt_total, int num_frames, float iou_threshold)
{
    EvalResult r;
    r.iou_threshold = iou_threshold;
    r.num_boards = static_cast<int>(per_board.size());
    r.num_frames = num_frames;
    r.totals = {0, 0, 0};

    for (const auto &m : per_board) {
        r.totals.tp += m.tp;
        r.totals.fp += m.fp;
        r.totals.fn += m.fn;
    }

    r.precision = safe_div(r.totals.tp, r.totals.tp + r.totals.fp);
    r.recall = safe_div(r.totals.tp, r.totals.tp + r.totals.fn);
    r.f1 = safe_div(2.0f * r.precision * r.recall, r.precision + r.recall);
    r.map50 = compute_map50(std::move(all_dets), num_gt_total);
    r.per_board = std::move(per_board);

    return r;
}

} // namespace evaluator
