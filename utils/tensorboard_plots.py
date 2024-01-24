import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


# Function to load tensorboard data
def load_tensorboard_data(logdir):
    event_files = [os.path.join(logdir, file) for file in os.listdir(logdir) if file.__contains__('events.out')]
    event_data = []
    for event_file in event_files:
        event_acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        data = {}

        for tag in tags:
            events = event_acc.Scalars(tag)
            steps, values = zip(*[(event.step, event.value) for event in events])
            data[tag] = pd.DataFrame({tag: values}, index=steps)
        event_data.append(pd.concat(data.values(), axis=1, join='inner'))

    return pd.concat(event_data, axis=1)


def extract_data_from_event_files(subdirectories, save_file_name="data_extract_summary", overwrite=False):
    for subdirectory in subdirectories:
        if not os.path.exists(os.path.join(subdirectory, f"{save_file_name}.pkl")) or overwrite:
            data = load_tensorboard_data(subdirectory)
            with open(f'{os.path.join(subdirectory, f"{save_file_name}.pkl")}', 'wb') as f:
                pickle.dump(data, f)


def load_tensorboard_data_from_pkl(pkl_path, file_name="data_extract_summary"):
    with open(os.path.join(pkl_path, f"{file_name}.pkl"), 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    # Folder containing tensorboard files
    folder_path = '/cache/PrefVeC_results_forv2/extra_algos/'
    saved_file_name = "data_extract_pd"
    timestep_key = "ray/tune/episodes_total"
    # List all subdirectories (assuming each subdirectory corresponds to a different model)
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if
                      os.path.isdir(os.path.join(folder_path, name))]
    extract_data_from_event_files(subdirectories, save_file_name=saved_file_name, overwrite=False)
    names_for_y_axes = {"ray/tune/evaluation/episode_reward_mean": "Mean Episode Reward",
                        "ray/tune/evaluation/episode_len_mean": "Mean Episode Length",
                        "ray/tune/evaluation/custom_metrics/success_mean": "Success Rate",
                        "ray/tune/evaluation/custom_metrics/slow_mean": "Slow Rate",
                        "ray/tune/evaluation/custom_metrics/collision_mean": "Collision Rate",
                        }
    data_from_subdirectories = {}
    # Initialize lists to store mean curves for each model
    for key in names_for_y_axes.keys():
        mean_curves = {}
        # Load tensorboard data for each model and store in a list
        for subdirectory in subdirectories:
            if subdirectory not in data_from_subdirectories.keys():
                data_from_subdirectories[subdirectory] = load_tensorboard_data_from_pkl(subdirectory, file_name=saved_file_name)
            if "grid" in subdirectory:
                mean_curves[subdirectory] = data_from_subdirectories[subdirectory]
        #
        # Plotting
        plt.figure(figsize=(10, 6))
        # Plot the mean curve for each model
        for model, dict_x in mean_curves.items():
            if dict_x[timestep_key].ndim < 2:
                dict_x = pd.concat([dict_x, dict_x], axis=1)
            mask = (dict_x[timestep_key] < 0.2e5).mean(axis=1) > 0
            dict_values = dict_x[key][mask]
            # Calculate mean curve
            mean_steps = dict_x[timestep_key][mask].mean(axis=1)
            mean_values = dict_values.mean(axis=1)
            std_values = dict_values.std(axis=1)
            min_values = dict_values.min(axis=1)
            max_values = dict_values.max(axis=1)
            # if std_values.sum() == 0:
            #     # add random noise to std_values to avoid error between 0 and 0.1
            #     noise = np.random.rand(std_values.shape[0]) * 0.1
            #     std_values += noise
            #     min_values -= noise
            #     max_values += noise

            # Plot shaded area for mean curve
            plt.fill_between(mean_steps, np.clip(mean_values - std_values, min_values, max_values),
                             np.clip(mean_values + std_values, min_values, max_values), alpha=0.3,
                             label=f'{os.path.split(model)[-1]}')

            # Plot mean curve
            plt.plot(mean_steps, mean_values)

        # Set plot title and labels
        plt.title('Averaged Training Curves with 10 different seeds')
        plt.xlabel(f'{timestep_key.split("/")[-1].replace("_", " ")} [-]')
        plt.ylabel(f'{names_for_y_axes[key]}')

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()
