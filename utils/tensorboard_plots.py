import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
        event_data.append(pd.concat(data.values(), axis=1))

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
    timestep_keys = ["ray/tune/episodes_total", ]
    # List all subdirectories (assuming each subdirectory corresponds to a different model)
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if
                      os.path.isdir(os.path.join(folder_path, name))]
    extract_data_from_event_files(subdirectories, save_file_name=saved_file_name, overwrite=False)

    # reverse key and value for names_for_y_axes
    names_for_y_axes = {"Mean Episode Reward": ["ray/tune/evaluation/episode_reward_mean"],
                        "Mean Episode Length": ["ray/tune/evaluation/episode_len_mean"],
                        "Success Rate": ["ray/tune/evaluation/custom_metrics/success_mean"],
                        "Slow Rate": ["ray/tune/evaluation/custom_metrics/slow_mean"],
                        "Collision Rate": ["ray/tune/evaluation/custom_metrics/collision_mean",
                                           "evaluation/collision_0"],
                        }
    data_from_subdirectories = {}

    mean_curves = {}
    # Load tensorboard data for each model and store in a list
    for subdirectory in subdirectories:
        if subdirectory not in data_from_subdirectories.keys():
            # if "grid" in subdirectory:
            data_from_subdirectories[subdirectory] = load_tensorboard_data_from_pkl(subdirectory,
                                                                                    file_name=saved_file_name)
    # Initialize lists to store mean curves for each model
    for plot_key, possible_keys in names_for_y_axes.items():
        # Plotting
        plt.figure(figsize=(10, 6))
        # Plot the mean curve for each model
        for model, dict_x in data_from_subdirectories.items():
            for key in possible_keys:
                if key in dict_x:
                    # Find the first matching timestep key or use default
                    time_steps = next(
                        (dict_x[timestep_key] for timestep_key in timestep_keys if timestep_key in dict_x),
                        pd.DataFrame(dict_x.index, columns=["step"], index=dict_x.index))

                    # Ensure time_steps and dict_x have 2 dimensions
                    if dict_x[key].ndim < 2:
                        time_steps = pd.concat([time_steps, time_steps], axis=1)
                        dict_x = pd.concat([dict_x, dict_x], axis=1)

                    # Apply mask for time_steps less than 0.2e5
                    mask = (time_steps < 0.2e5).mean(axis=1) > 0
                    dict_values = dict_x[key][mask]

                    # Calculate statistics
                    mean_steps = time_steps[mask].mean(axis=1)
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
        plt.xlabel(f'{timestep_keys[0].split("/")[-1].replace("_", " ")} [-]')
        plt.ylabel(f'{plot_key}')

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()
