import numpy as np
import pandas as pd
import cv2
import random
import optuna

from unreal_utils_advanced import run_unreal_executable
from detection_utils_advanced import run_detection

from optuna.samplers import TPESampler
from optuna.integration import TensorBoardCallback

import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go


def customize_figure_history(fig):
    # Change the font type
    fig.update_layout(font=dict(family='CMU Serif'))

    fig.data[0].update(
        line=dict(color='rgba(146, 164, 0, 1)', width=1.5),
        marker=dict(size=10)
    )
    fig.data[1].update(
        line=dict(color='rgba(93, 101, 143, 1)', width=3),
        marker=dict(size=10)
    )

    # Update legend
    fig.update_layout(
        title_font=dict(color='black'),
        xaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
            linewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
        yaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
            linewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
        font=dict(family='CMU Serif', size=14),
        legend=dict(
            x=1.0,
            y=1.0,
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            traceorder='normal'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
    )
    return fig

def customize_figure_importance(fig):
    # Change the font type
    fig.update_layout(font=dict(family='CMU Serif'))

    fig.data[0].update(
        marker=dict(color='rgba(14, 24, 86, 1)'),
    )

    # Update legend
    fig.update_layout(
        title_font=dict(color='black'),
        xaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
            linewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
        yaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
            linewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
        font=dict(family='CMU Serif', size=14),
        legend=dict(
            x=1.0,
            y=1.0,
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            traceorder='normal'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
    )
    return fig

def plot_accuracies(study):
    ped_accuracies = [trial.user_attrs["pedestrian_accuracies"] for trial in study.trials]
    car_accuracies = [trial.user_attrs["car_accuracies"] for trial in study.trials]

    ped_colors = [
    "rgba(14, 24, 86, 1)",
    "rgba(146, 164, 0, 1)",
    "rgba(0, 255, 0, 1)",
    "rgba(255, 0, 0, 1)",
    "rgba(0, 0, 255, 1)",
    "rgba(255, 255, 0, 1)",
    "rgba(255, 0, 255, 1)",
    "rgba(0, 255, 255, 1)",
    "rgba(128, 0, 0, 1)",
    "rgba(0, 128, 0, 1)",
    "rgba(0, 0, 128, 1)",
]  
    car_colors = [
        "rgba(93, 101, 143, 1)", 
        "red",
        "rgba(128, 0, 128, 1)",
        "rgba(128, 128, 0, 1)",
        "rgba(0, 128, 128, 1)",
        "rgba(128, 128, 128, 1)",
        "rgba(255, 165, 0, 1)",
        "rgba(210, 105, 30, 1)",
        "rgba(0, 128, 0, 1)",
        "rgba(128, 0, 0, 1)",
        "rgba(0, 0, 255, 1)",
        "rgba(255, 0, 255, 1)",
        "rgba(255, 255, 0, 1)",
        "rgba(255, 192, 203, 1)",
        "blue",
    ]  


    fig = go.Figure()

    # Plot pedestrian accuracies
    for pos_idx in range(len(ped_accuracies[0])):
        fig.add_trace(
            go.Scatter(
                x=list(range(0, len(ped_accuracies))),
                y=[acc[pos_idx] for acc in ped_accuracies],
                mode="markers+lines",
                name=f"Pedestrian Accuracy (Pos {pos_idx+1})",
                line=dict(color=ped_colors[pos_idx], width=3)
            )
        )

    # Plot car accuracies
    for pos_idx in range(len(car_accuracies[0])):
        fig.add_trace(
            go.Scatter(
                x=list(range(0, len(car_accuracies))),
                y=[acc[pos_idx] for acc in car_accuracies],
                mode="markers+lines",
                name=f"Car Accuracy (Pos {pos_idx+1})",
                line=dict(color=car_colors[pos_idx], width=3)
            )
        )

    fig.update_layout(
        title="Detection Accuracies per Trial",
        xaxis_title="Trial",
        yaxis_title="Accuracy",
        title_font=dict(color='black'),
        xaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
            linewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
        yaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            gridcolor='rgba(128, 128, 128, 0.5)',
            gridwidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
            linewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=1,
        ),
        font=dict(family='CMU Serif', size=14),
        legend=dict(
            x=1.0,
            y=1.0,
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            traceorder='normal'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
    )
    return fig

class SaveStudyCallback:
    def __init__(self):
        pass

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        i = trial.number
        if i % 50 == 0:
            joblib.dump(study, f"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/Study_{i}.pkl")

class SaveBestResultCallback:
    def __init__(self):
        pass

    def __call__(self, study, trial):
        if study.best_trial == trial:
            i =trial.number 
            df_result = pd.read_csv(
                r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/object_detection/Object_Detection_Results_Advanced.csv"
            )
            df_result.to_csv(f"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/Object_Detection_Best_Results_{i}.csv", index=False)


def objective(trial):
    # Best parameters from previous
    ld1 = 0.5715755083683963
    ld2 = 0.14314837116075532
    ld3 = 0.46249187895343696

    # Create the four 4x4 arrays with the optimized values
    arr1 = np.full((32, 32), ld1)
    arr2 = np.full((32, 32), ld2)
    arr3 = np.full((32, 32), ld3)

    # Best parameters from previous
    ld2_it1 = 0.012456362349803081
    ld3_it1 = 0.28014691098047945
    ld4_it1 = 0.27578502174243374

    # Create the four 4x4 arrays with the optimized values
    arr2_it1 = np.full((16, 16), ld2_it1) 
    arr3_it1 = np.full((16, 16), ld3_it1) 
    arr4_it1 = np.full((16, 16), ld4_it1) 

    # Define the light distribution as a set of hyperparameters
    ld1_it2 = trial.suggest_uniform("ld1_it2", 0, 1)
    ld2_it2 = trial.suggest_uniform("ld2_it2", 0, 1)
    ld3_it2 = trial.suggest_uniform("ld3_it2", 0, 1)
    ld4_it2 = trial.suggest_uniform("ld4_it2", 0, 1)
    
    # Create the four 4x4 arrays with the optimized values
    arr1_it2 = np.full((8, 8), ld1_it2)
    arr2_it2 = np.full((8, 8), ld2_it2)
    arr3_it2 = np.full((8, 8), ld3_it2)
    arr4_it2 = np.full((8, 8), ld4_it2)

    # Create arr_it2
    arr_it2 = np.concatenate([np.concatenate([arr1_it2, arr2_it2], axis=1), np.concatenate([arr3_it2, arr4_it2], axis=1)], axis=0)

    # Concatenate the four 4x4 arrays into an 8x8 array
    arr4 = np.concatenate(
        [np.concatenate([arr_it2, arr2_it1], axis=1), np.concatenate([arr3_it1, arr4_it1], axis=1)], axis=0
    )

    # Concatenate the four 4x4 arrays into an 8x8 array
    light_distribution = np.concatenate(
        [np.concatenate([arr1, arr2], axis=1), np.concatenate([arr3, arr4], axis=1)], axis=0
    )

    # Convert numpy array to png file
    cv2.imwrite(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/Light_distribution_rechts.png",
        light_distribution * 255,
    )
    cv2.imwrite(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/Light_distribution_links.png",
        light_distribution * 255,
    )

    # Run the unreal engine executable
    run_unreal_executable(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/unreal_engine/UE_Light_Sim_Screenshot_Advanced_Exe/Windows/UE_Light_Sim.exe"
    )
    # Run detection on the generated images
    run_detection()

    # Get detection accuracies and RMS values
    ped_acc, car_acc, ped_RMS_mean, car_RMS_mean = get_accuracy()

    ###### Accuracies Figure #####
    trial.set_user_attr("pedestrian_accuracies", ped_acc.tolist())
    trial.set_user_attr("car_accuracies", car_acc.tolist())
    ###### Accuracies Figure #####

    # List of Positions
    ped_pos_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    car_pos_list = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]

    # Weights for positions
    ### Ped
    # max_weight = 1 min_weight = 0.5
    # Weights_list = [(max_weight - min_weight) * i/(len(ped_pos_list)-1) + min_weight for i in range(len(ped_pos_list))]
    ### Car
    # Find the range of the list # min_pos = min(car_pos_list) # max_pos = max(car_pos_list) # pos_range = max_pos - min_pos
    # # Normalize the values and apply the linear scaling
    # weights = [1 - 0.5 * (pos - min_pos) / pos_range for pos in car_pos_list]
    
    W_ped_pos = [1.0, 0.95, 0.91, 0.86, 0.82, 0.77, 0.73, 0.68, 0.64, 0.59, 0.55]
    W_car_pos = [1.0, 0.96, 0.93, 0.89, 0.86, 0.82, 0.79, 0.75, 0.71, 0.68, 0.64, 0.61, 0.57, 0.54, 0.5]

    # Weights for ped and car
    W_ped = 1
    W_car = 0.8

    # Declare total_acc
    total_err_acc = 0
    total_ped_err_acc = 0
    total_car_err_acc = 0

    for i in range(len(ped_pos_list)):
        # Sum the accuracy
        total_ped_err_acc += W_ped * W_ped_pos[i] * (1 - ped_acc[i])
    
    for i in range(len(car_pos_list)):
        # Sum the accuracy
        total_car_err_acc += W_car * W_car_pos[i] * (1 - car_acc[i])

    total_err_acc = total_ped_err_acc + total_car_err_acc
    print("total_ped_err_acc", total_ped_err_acc)
    print("total_car_err_acc", total_car_err_acc)

    # Add penalty for RMS
    penalty_rms = np.sum(ped_RMS_mean) + np.sum(car_RMS_mean)
    
    # Add a penalty term to discourage the use of high values for the light distribution
    penalty_light = light_distribution.mean()

    # Sum the penalties
    total_penalty = penalty_rms + penalty_light

    # Set the total penalty value as a user attribute for the trial
    trial.set_user_attr("penalty_rms", penalty_rms)
    trial.set_user_attr("penalty_light", penalty_light)
    trial.set_user_attr("total_penalty * 10%", total_penalty*0.1)

    # Return the total error accuracy plus the penalty term as the objective function
    return total_err_acc + total_penalty*0.1


def get_accuracy():
    df_result = pd.read_csv(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/object_detection/Object_Detection_Results_Advanced.csv"
    )

    # calculate the mean confidence score for each combination of distance and class
    df_result["mean_confidence_score"] = df_result.groupby(["distance", "class"])[
        "confidence_score"
    ].transform("mean")

    # calculate the root-mean-square (RMS) error for each row in the DataFrame
    df_result["RMS"] = np.sqrt(
        np.absolute(df_result["confidence_score"] - df_result["mean_confidence_score"]) ** 2
    )

    # round the RMS column to 6 decimal places
    df_result["RMS"] = df_result["RMS"].round(6)

    # calculate the mean RMS error for each combination of distance and class
    mean_rms = df_result.groupby(["distance", "class"])["RMS"].mean().reset_index()

    # merge the mean RMS error back into the original DataFrame
    df_result = df_result.merge(mean_rms, on=["distance", "class"], suffixes=("", "_mean"))

    # print the final DataFrame
    print(df_result)

    # calculate the mean confidence score for pedestrian and car classes at each distance
    pedestrian_confidence = (
        df_result[df_result["class"] == "pedestrian"]
        .groupby("distance")["confidence_score"]
        .mean()
    )

    car_confidence = (
        df_result[df_result["class"] == "car"]
        .groupby("distance")["confidence_score"]
        .mean()
    )

    # convert the mean confidence scores to numpy arrays
    ped_arr = np.array(pedestrian_confidence.values, dtype=np.float32)
    car_arr = np.array(car_confidence.values, dtype=np.float32)

    # print the mean confidence scores for pedestrian and car classes at each distance
    print("ped_acc_arr", ped_arr)
    print("car_acc_arr", car_arr)

    # calculate the mean RMS error for pedestrian and car classes at each distance
    ped_RMS_mean = (
        df_result[df_result["class"] == "pedestrian"].groupby("distance")["RMS_mean"].mean()
    )

    car_RMS_mean = (
        df_result[df_result["class"] == "car"].groupby("distance")["RMS_mean"].mean()
    )

    # convert the mean RMS errors to numpy arrays
    ped_RMS_mean_arr = np.array(ped_RMS_mean.values, dtype=np.float32)
    car_RMS_mean_arr = np.array(car_RMS_mean.values, dtype=np.float32)

    # print the mean RMS errors for pedestrian and car classes at each distance
    #print(ped_RMS_mean_arr, car_RMS_mean_arr)

    return ped_arr, car_arr, ped_RMS_mean_arr, car_RMS_mean_arr


if __name__ == "__main__":
    #log_dir = r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/TB_Log"

    study = optuna.create_study(sampler=TPESampler(n_startup_trials=100))
    study.optimize(
        objective,
        n_trials=500,
        #callbacks=[TensorBoardCallback(log_dir, metric_name="light_opt")],
        callbacks=[SaveStudyCallback(), SaveBestResultCallback()]
    )


    print("Best light distribution: ", study.best_params)
    print("Best accuracy: ", study.best_value)

    # Save the study using joblib 
    joblib.dump(study, "C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/Study.pkl")

    # Plot the accuracies
    fig_acc = plot_accuracies(study)
    
    # Plot the graphs
    fig_hist = optuna.visualization.plot_optimization_history(study)
    fig_hist = customize_figure_history(fig_hist)

    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance = customize_figure_importance(fig_importance)
    
    # Write images
    fig_acc.write_image("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/Graphs/fig_acc.svg")
    fig_hist.write_image("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/Graphs/fig_hist.svg")
    fig_importance.write_image("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies_advanced/Study_Level_2_Startup/Graphs/fig_importance.svg")
    
    # Show images
    fig_acc.show()
    fig_hist.show()
    fig_importance.show()

