from argparse import ArgumentParser

from light_utils import csv_to_png
from unreal_utils import run_unreal_executable
from detection_utils import run_detection


def parse_args():
    # create the parser
    parser = ArgumentParser()
    # add the arguments
    parser.add_argument(
        "--csv_file",
        default=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_csv/Beamer_Optoma_links_2013v3.csv",
        help="path csv file",
    )
    parser.add_argument(
        "--png_file",
        default=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/",
        help="path png file",
    )
    parser.add_argument(
        "--exe_dir",
        default=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/unreal_engine/UE_Light_Sim_Exe/Windows/UE_Light_Sim.exe",
        help="path to unreal simulation executable",
    )
    # parse the arguments
    args = parser.parse_args()
    return args


def main(args):
    """
    Main function for the Light Distribution Optimization project:

    This function uses the argparse module to parse functions arguments,
    the "csv_to_png" function from the light_utils module to convert CSV files to PNG images,
    the "run_unreal_executable" function from the unreal_utils module to run an Unreal Engine executable,
    and the "run_detection" function from the detection_utils module to perform detection tasks.
    """
    # Load the csv_file and generate two png files for right and left headlamp
    #csv_to_png(args.csv_file, args.png_file)
    # Run the Unreal environment executable which will generate the scenario and render the images
    #run_unreal_executable(args.exe_dir)
    # Run the detection on the generated images using an Object Detection Algorithm (default=Faster R-CNN)
    # The path of the rendered images can be changed in "object_detection/Detection.py"
    run_detection()


if __name__ == "__main__":
    args = parse_args()
    main(args)
