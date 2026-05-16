#include "board_inference.hpp"
#include "frame_loader.hpp"
#include "io.hpp"
#include "yolo_session.hpp"

#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("wood_knot_detector", "1.0");
    program.add_description("Wood knot detection from sequential board frame images.");

    argparse::ArgumentParser predict_cmd("predict");
    predict_cmd.add_description("Run knot detection on a directory of board frames.");
    predict_cmd.add_argument("--frames-dir")
        .required()
        .help("Directory containing {board}_{frame}.png files");
    predict_cmd.add_argument("--model").required().help("Path to ONNX model");
    predict_cmd.add_argument("--out").required().help("Output directory for per-board JSON files");
    predict_cmd.add_argument("--device")
        .default_value(std::string("cpu"))
        .help("Inference device: cpu | gpu");
    predict_cmd.add_argument("--conf-threshold")
        .default_value(0.25f)
        .scan<'f', float>()
        .help("Minimum confidence score to keep a detection");

    argparse::ArgumentParser test_cmd("test");
    test_cmd.add_description("Run knot detection and evaluate against ground-truth labels.");
    test_cmd.add_argument("--frames-dir")
        .required()
        .help("Directory containing {board}_{frame}.png files");
    test_cmd.add_argument("--labels-dir")
        .required()
        .help("Directory containing {board}_{frame}.txt YOLO labels");
    test_cmd.add_argument("--model").required().help("Path to ONNX model");
    test_cmd.add_argument("--out")
        .required()
        .help("Output directory for predictions, metrics.json and REPORT.md");
    test_cmd.add_argument("--device")
        .default_value(std::string("cpu"))
        .help("Inference device: cpu | gpu");
    test_cmd.add_argument("--conf-threshold")
        .default_value(0.25f)
        .scan<'f', float>()
        .help("Minimum confidence score to keep a detection");
    test_cmd.add_argument("--iou-threshold")
        .default_value(0.5f)
        .scan<'f', float>()
        .help("IoU threshold for TP/FP matching in evaluation");

    program.add_subparser(predict_cmd);
    program.add_subparser(test_cmd);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n\n" << program;
        return 1;
    }

    if (program.is_subcommand_used("predict")) {
        auto& cmd          = program.at<argparse::ArgumentParser>("predict");
        const auto frames_dir = cmd.get("--frames-dir");
        const auto model      = cmd.get("--model");
        const auto out        = cmd.get("--out");
        const auto device     = cmd.get("--device");
        const float conf_thr  = cmd.get<float>("--conf-threshold");

        const auto boards = frame_loader::build_boards(frames_dir);
        YoloSession session(model, device);

        const std::filesystem::path out_dir(out);
        int board_num = 0;
        for (const auto& [board_id, frame_paths] : boards) {
            auto result = board_inference::predict_board(frame_paths, session, conf_thr);
            io::write_board_json(result, out_dir);
            std::cout << "board " << board_id << ": " << result.knots.size() << " knots\n";
            ++board_num;
        }
        std::cout << "Done. " << board_num << " boards → " << out << '\n';
        return 0;
    }

    if (program.is_subcommand_used("test")) {
        std::cout << "test mode — not yet implemented\n";
        return 0;
    }

    std::cerr << program;
    return 1;
}
