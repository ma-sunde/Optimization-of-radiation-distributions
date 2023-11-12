import numpy as np
import pandas as pd
import cv2
import random
import optuna

from unreal_utils import run_unreal_executable
from detection_utils import run_detection

from optuna.samplers import TPESampler, CmaEsSampler
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

    ped_colors = ['rgba(14, 24, 86, 1)', 'rgba(146, 164, 0, 1)']  # Specify colors for pedestrian accuracies
    car_colors = ['rgba(93, 101, 143, 1)', 'red']  # Specify colors for car accuracies

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



def objective(trial):
    # Define the light distribution as a set of hyperparameters
    light_distribution = [
        trial.suggest_categorical(f"LI_{i}", [0, 1]) for i in range(64)
    ]
    # print(light_distribution)

    # Convert list to numpy array
    light_distribution_np_array = np.array(light_distribution)

    # Convert sample to png file
    cv2.imwrite(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/Light_distribution_rechts.png",
        light_distribution_np_array.reshape(8, 8) * 255,
    )
    cv2.imwrite(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/Light_distribution_links.png",
        light_distribution_np_array.reshape(8, 8) * 255,
    )

    # Run the unreal engine executable
    run_unreal_executable(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/unreal_engine/UE_Light_Sim_Exe/Windows/UE_Light_Sim.exe"
    )
    # Run detection on the generated images
    run_detection()
    # Get detection accuracies
    ped_acc, car_acc = get_accuracy()

    ###### Accuracies Figure #####
    trial.set_user_attr("pedestrian_accuracies", ped_acc.tolist())
    trial.set_user_attr("car_accuracies", car_acc.tolist())
    ###### Accuracies Figure #####

    # List of Positions
    pos_list = [10, 20]

    # Weights for ped and car
    W_ped = 1
    W_car = 0.8
    # Weights for positions
    W_pos = [1, 0.8]

    # Declare total_acc
    total_err_acc = 0

    for i in range(len(pos_list)):
        # Sum the accuracy
        total_err_acc += W_ped * W_pos[i] * (1 - ped_acc[i]) + W_car * W_pos[i] * (
            1 - car_acc[i]
        )

    # Return the total accuracy as the objective function
    return total_err_acc


def get_accuracy():
    df_result = pd.read_csv(
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/object_detection/Object_Detection_Results.csv"
    )

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

    ped_arr = np.array(pedestrian_confidence.values, dtype=np.float32)
    car_arr = np.array(car_confidence.values, dtype=np.float32)

    return ped_arr, car_arr


if __name__ == "__main__":
    #log_dir = r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies/Study_4/TB_Log"

    study = optuna.create_study(sampler=TPESampler())
    study.optimize(
        objective,
        n_trials=1500,
        #callbacks=[TensorBoardCallback(log_dir, metric_name="light_opt")],
    )

    print("Best light distribution: ", study.best_params)
    print("Best accuracy: ", study.best_value)

    # Save the study using joblib
    joblib.dump(study, "C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies/Study_5/Study.pkl")

    # Resume the study from the saved file
    #loaded_study = joblib.load("test_stud_save.pkl")

    # Plot the accuracies
    fig_acc = plot_accuracies(study)
    
    # Plot the graphs
    fig_hist = optuna.visualization.plot_optimization_history(study)
    fig_hist = customize_figure_history(fig_hist)

    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance = customize_figure_importance(fig_importance)
    
    # Write images
    fig_acc.write_image("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies/Study_5/Graphs/fig_acc.svg")
    fig_hist.write_image("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies/Study_5/Graphs/fig_hist.svg")
    fig_importance.write_image("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/optuna/studies/Study_5/Graphs/fig_importance.svg")
    
    # Show images
    fig_acc.show()
    fig_hist.show()
    fig_importance.show()









    

