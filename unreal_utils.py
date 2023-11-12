import subprocess
import time
import os


def run_unreal_executable(executable_dir):
    # Open the text file and read the contents
    with open("unreal_engine/Execution_Command_with_Args.txt", "r") as f:
        execution_command_string = f.read()

    # Split the execution command string into a list of arguments
    command_list = execution_command_string.split(" ")
    # Insert the path to the executable at the beginning of the list
    command_list.insert(0, executable_dir)

    print("Simulation started at:" + time.strftime("%Y-%m-%d %H:%M:%S"))
    # Run the simulation using the subprocess module
    process = subprocess.Popen(command_list)
    # Wait for the simulation to finish
    process.wait()
    print("Simulation ended at:" + time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    executable_dir = r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/unreal_engine/UE_Light_Sim_Exe/Windows/UE_Light_Sim.exe"
    run_unreal_executable(executable_dir)
