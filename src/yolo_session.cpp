#include "yolo_session.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

static constexpr int kInputSize = 640;
static constexpr int kMaxDets = 300;

// ---------------------------------------------------------------------------
// Letterbox helpers
// ---------------------------------------------------------------------------
struct LetterboxInfo {
    cv::Mat canvas; // kInputSize x kInputSize, BGR uint8
    float scale;
    int pad_left;
    int pad_top;
};

static LetterboxInfo letterbox(const cv::Mat &bgr)
{
    const float scale = std::min(static_cast<float>(kInputSize) / bgr.cols,
                                 static_cast<float>(kInputSize) / bgr.rows);
    const int new_w = static_cast<int>(bgr.cols * scale);
    const int new_h = static_cast<int>(bgr.rows * scale);
    const int pad_left = (kInputSize - new_w) / 2;
    const int pad_top = (kInputSize - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(kInputSize, kInputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_left, pad_top, new_w, new_h)));

    return {canvas, scale, pad_left, pad_top};
}

static std::vector<float> mat_to_chw(const cv::Mat &bgr_canvas)
{
    cv::Mat rgb;
    cv::cvtColor(bgr_canvas, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> chw(3 * kInputSize * kInputSize);
    const int plane = kInputSize * kInputSize;
    for (int h = 0; h < kInputSize; ++h) {
        for (int w = 0; w < kInputSize; ++w) {
            const auto &px = rgb.at<cv::Vec3b>(h, w);
            chw[0 * plane + h * kInputSize + w] = px[0] / 255.0f;
            chw[1 * plane + h * kInputSize + w] = px[1] / 255.0f;
            chw[2 * plane + h * kInputSize + w] = px[2] / 255.0f;
        }
    }
    return chw;
}

// ---------------------------------------------------------------------------
// YoloSession::Impl
// ---------------------------------------------------------------------------
struct YoloSession::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    static Ort::Session make_session(Ort::Env &env, const std::filesystem::path &path,
                                     const std::string &device)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (device == "gpu") {
            OrtCUDAProviderOptions cuda_opts{};
            try {
                opts.AppendExecutionProvider_CUDA(cuda_opts);
            } catch (const Ort::Exception &e) {
                std::cerr << "[warn] CUDA EP unavailable: " << e.what()
                          << " — falling back to CPU\n";
            }
        }

        return Ort::Session(env, path.c_str(), opts);
    }

    Impl(const std::filesystem::path &path, const std::string &device)
        : env(ORT_LOGGING_LEVEL_WARNING, "wood_knot_detector"),
          session(make_session(env, path, device))
    {
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    }
};

// ---------------------------------------------------------------------------
// YoloSession public API
// ---------------------------------------------------------------------------
YoloSession::YoloSession(const std::filesystem::path &model_path, const std::string &device)
    : impl_(std::make_unique<Impl>(model_path, device))
{
}

YoloSession::~YoloSession() = default;

std::vector<Detection> YoloSession::predict(const cv::Mat &bgr_frame, float conf_threshold) const
{
    const auto lb = letterbox(bgr_frame);
    const auto chw = mat_to_chw(lb.canvas);

    const std::array<int64_t, 4> input_shape = {1, 3, kInputSize, kInputSize};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor =
        Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(chw.data()), chw.size(),
                                        input_shape.data(), input_shape.size());

    const char *in_name = impl_->input_names[0].c_str();
    const char *out_name = impl_->output_names[0].c_str();
    auto outputs =
        impl_->session.Run(Ort::RunOptions{nullptr}, &in_name, &input_tensor, 1, &out_name, 1);

    const float *data = outputs[0].GetTensorData<float>();

    std::vector<Detection> dets;
    dets.reserve(32);

    const float orig_w = static_cast<float>(bgr_frame.cols);
    const float orig_h = static_cast<float>(bgr_frame.rows);

    for (int i = 0; i < kMaxDets; ++i) {
        const float *row = data + i * 6;
        const float conf = row[4];
        if (conf < conf_threshold)
            continue;

        // Unscale from letterbox space to original frame space
        float x1 = (row[0] - lb.pad_left) / lb.scale;
        float y1 = (row[1] - lb.pad_top) / lb.scale;
        float x2 = (row[2] - lb.pad_left) / lb.scale;
        float y2 = (row[3] - lb.pad_top) / lb.scale;

        x1 = std::clamp(x1, 0.0f, orig_w);
        y1 = std::clamp(y1, 0.0f, orig_h);
        x2 = std::clamp(x2, 0.0f, orig_w);
        y2 = std::clamp(y2, 0.0f, orig_h);

        if (x2 <= x1 || y2 <= y1)
            continue;

        dets.push_back({x1, y1, x2, y2, conf, static_cast<int>(row[5])});
    }

    return dets;
}
