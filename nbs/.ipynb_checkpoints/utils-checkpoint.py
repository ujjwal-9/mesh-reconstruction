import os
import argparse
from utils import main
from configs import config, thresholds


def get_args(reconstruction_method, file_path):
    file = os.path.basename(file_path).split(".")[0]
    config["reconstruction_method"] = reconstruction_method
    config[reconstruction_method]["min_density_threshold"] = thresholds[file]
    return config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Point Cloud Reconstruction Tool")

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input point cloud file (PLY format)",
    )

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="poisson",
        choices=["poisson", "hole_preserve"],  # Add other methods as needed
        help="Reconstruction method to use (default: poisson)",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save output files (default: outputs)",
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Prepare configuration
    reconstruction_config = get_args(args.method, args.input_file)

    # Perform reconstruction
    mesh = main(args.input_file, args.output_dir, **reconstruction_config)

    return mesh


if __name__ == "__main__":
    main()
