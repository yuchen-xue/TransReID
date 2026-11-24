import matplotlib.pyplot as plt
import numpy as np


def plot_matches(match_array: np.ndarray, fig_path: str = "matches_visualization.png"):

    # Plot the matches
    plt.figure(figsize=(10, 6))
    # Assuming arr_matches is a 2D array where each row represents a match
    plt.imshow(match_array, cmap="Pastel1", aspect="auto")
    plt.colorbar(label="Match Score")
    plt.title("Visualization of Matches")
    plt.xlabel("Gallery Images")
    plt.ylabel("Query Images")
    # Save the plot to a file
    plt.savefig(fig_path)


def plot_distance_matrix(
    distmat_array: np.ndarray, fig_path: str = "distance_matrix_visualization.png"
):

    # Plot the distance matrix
    plt.figure(figsize=(10, 6))
    # Assuming arr_distmat is a 2D array representing the distance matrix
    plt.imshow(distmat_array, cmap="viridis", aspect="auto")
    plt.colorbar(label="Distance")
    plt.title("Visualization of Distance Matrix")
    plt.xlabel("Gallery Images")
    plt.ylabel("Query Images")
    # Save the plot to a file
    plt.savefig(fig_path)


if __name__ == "__main__":
    arr_matches = np.load("matches-refined_manual.npy", allow_pickle=True)
    arr_distmat = np.load("distmat-refined_manual.npy", allow_pickle=True)

    # arr_matches = np.load('matches.npy', allow_pickle=True)
    # arr_distmat = np.load('distmat.npy', allow_pickle=True)

    plot_matches(arr_matches, fig_path="matches_visualization.png")
    plot_distance_matrix(arr_distmat, fig_path="distance_matrix_visualization.png")
