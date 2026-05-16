#include <argparse/argparse.hpp>
#include <iostream>
#include <string>

static argparse::ArgumentParser make_predict_parser()
{
    argparse::ArgumentParser cmd("predict");
    cmd.add_description("Run knot detection on a directory of board frames.");
    cmd.add_argument("--frames-dir").required().help("Directory containing {board}_{frame}.png files");
    cmd.add_argument("--model").required().help("Path to ONNX model");
    cmd.add_argument("--out").required().help("Output directory for per-board JSON files");
    cmd.add_argument("--device").default_value(std::string("cpu")).help("Inference device: cpu | gpu");
    cmd.add_argument("--conf-threshold")
        .default_value(0.25f)
        .scan<'f', float>()
        .help("Minimum confidence score to keep a detection");
    return cmd;
}

static argparse::ArgumentParser make_test_parser()
{
    argparse::ArgumentParser cmd("test");
    cmd.add_description("Run knot detection and evaluate against ground-truth labels.");
    cmd.add_argument("--frames-dir").required().help("Directory containing {board}_{frame}.png files");
    cmd.add_argument("--labels-dir").required().help("Directory containing {board}_{frame}.txt YOLO labels");
    cmd.add_argument("--model").required().help("Path to ONNX model");
    cmd.add_argument("--out").required().help("Output directory for predictions, metrics.json and REPORT.md");
    cmd.add_argument("--device").default_value(std::string("cpu")).help("Inference device: cpu | gpu");
    cmd.add_argument("--conf-threshold")
        .default_value(0.25f)
        .scan<'f', float>()
        .help("Minimum confidence score to keep a detection");
    cmd.add_argument("--iou-threshold")
        .default_value(0.5f)
        .scan<'f', float>()
        .help("IoU threshold for TP/FP matching in evaluation");
    return cmd;
}

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("wood_knot_detector", "1.0");
    program.add_description("Wood knot detection from sequential board frame images.");

    auto predict_cmd = make_predict_parser();
    auto test_cmd = make_test_parser();

    program.add_subparser(predict_cmd);
    program.add_subparser(test_cmd);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n\n" << program;
        return 1;
    }

    if (program.is_subcommand_used("predict")) {
        std::cout << "predict mode — not yet implemented\n";
        return 0;
    }

    if (program.is_subcommand_used("test")) {
        std::cout << "test mode — not yet implemented\n";
        return 0;
    }

    std::cerr << program;
    return 1;
}
