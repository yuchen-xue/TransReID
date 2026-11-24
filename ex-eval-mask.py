from pathlib import Path

import numpy as np
from numpy import ma
from numpy.typing import NDArray


def get_cmc_curve(matches: NDArray[np.bool_]) -> NDArray[np.float64]:


    # Compute top-k accuracy for each query
    cumsum_matches = matches.cumsum(axis=1)
    cumsum_matches[cumsum_matches > 1] = 1
    top_k_acc = cumsum_matches

    # Compute CMC (Cumulative Matching Characteristic) curve
    cmc = top_k_acc.mean(axis=0)
    return cmc



def get_average_precision(matches: NDArray[np.bool_], exclude_mask: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:

    # Compute top-k precision for each query
    cumsum_matches = matches.cumsum(axis=1)
    rank_position = np.arange(1, cumsum_matches.shape[1] + 1)
    if exclude_mask is not None:
        print(f"Exclude mask:\n {exclude_mask.astype(np.int8)}")
        # Apply the exclude mask to the cumsum_matches and rank_position
        cumsum_matches = ma.masked_array(cumsum_matches, mask=exclude_mask)
        rank_position = exclude_mask.cumsum(axis=1)
        rank_position = ma.masked_array(rank_position, mask=exclude_mask)
        print(f"Rank position after applying exclude_mask:\n {rank_position}")
    top_k_precision = cumsum_matches / rank_position
    # print(f"{top_k_precision.shape=}, {top_k_precision.dtype=}")
    print(f"Overview of Top-k Precision:\n {top_k_precision[:5, :10]}")  # Print first 5 queries and first 10 ranks

    # Compute Average Precision (AP) for each query
    if exclude_mask is not None:
        # Apply the exclude mask to the matches
        matches = ma.masked_array(matches, mask=exclude_mask)
        top_k_precision = ma.masked_array(top_k_precision, mask=exclude_mask)
    matched_cum_prec = top_k_precision * matches
    # if exclude_mask is not None:
    #     # Apply the exclude mask to the matched_cum_prec
    #     matched_cum_prec = ma.masked_array(matched_cum_prec, mask=exclude_mask)
    if exclude_mask is not None:
        # Apply the exclude mask to the matched_cum_prec
        matches = ma.masked_array(matches, mask=exclude_mask)
    num_rel = matches.sum(axis=1)
    if exclude_mask is not None:
        # Apply the exclude mask to the num_rel
        matched_cum_prec = ma.masked_array(matched_cum_prec, mask=exclude_mask)
    average_precision = matched_cum_prec.sum(axis=1) / num_rel
    return average_precision


def get_cmc_mAP(
    dist_mat: NDArray[np.float64],
    query_pid_arr: NDArray[np.int_],
    gallery_pid_arr: NDArray[np.int_],
    exclude_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate the CMC curve and mean Average Precision (mAP) given a distance matrix and corresponding PIDs.

    Args:
        dist_mat: Distance matrix of shape [num_queries, num_gallery]
        query_pid_arr: Array of query PIDs of shape [num_queries]
        gallery_pid_arr: Array of gallery PIDs of shape [num_gallery]
        exclude_mask: Optional boolean mask to exclude certain matches
    """

    if exclude_mask is not None:
        # Apply the exclude mask to the distance matrix
        dist_mat = ma.masked_array(dist_mat, mask=exclude_mask)
        # print(f"Masked distance matrix:\n {dist_mat}")
    
    # Get the indices that would sort the distance matrix by row
    argsort_distmat = np.argsort(dist_mat, axis=1)

    # Create a boolean mask for matching PIDs
    pid_matches: NDArray[np.bool_] = (
        gallery_pid_arr[np.newaxis, :] == query_pid_arr[:, np.newaxis]
    )

    # Get sorted pid_matches based on the argsort of the distance matrix
    sorted_pid_matches: NDArray[np.bool_] = pid_matches[
        np.arange(pid_matches.shape[0])[:, np.newaxis], argsort_distmat
    ]

    cmc = get_cmc_curve(sorted_pid_matches)
    print(f"CMC curve:\n {cmc}")
    # print(f"CMC curve:\n {cmc[:10]}")  # Print first 10 ranks

    # Compute mean Average Precision (mAP)
    average_precision = get_average_precision(sorted_pid_matches, exclude_mask)

    return cmc, average_precision


# distmat = np.array(
#     [
#         [0.1, 0.8, 0.2, 0.3, 0.5, 0.4, 0.7, 0.6],
#         [0.5, 0.1, 0.3, 0.2, 0.4, 0.6, 0.8, 0.7],
#         [0.2, 0.3, 0.1, 0.4, 0.6, 0.5, 0.7, 0.8],
#         [0.3, 0.2, 0.4, 0.1, 0.5, 0.6, 0.8, 0.7],
#         [0.4, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.8],
#         [0.6, 0.7, 0.8, 0.5, 0.2, 0.1, 0.4, 0.3],
#         [0.7, 0.6, 0.5, 0.8, 0.3, 0.4, 0.1, 0.2],
#         [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
#     ]
# )

# q_pids = np.array([1, 1, 1, 2, 2, 2, 2, 2])
# g_pids = np.array([2, 1, 1, 2, 2, 2, 1, 1])

# input_dir = Path("output/Test-output/Visioon-refined_manual")
input_dir = Path("output/Test-output/Visioon-refined_manual-new")
distmat = np.load(input_dir / "distmat.npy")
q_pids = np.load(input_dir / "q_pids.npy")
g_pids = np.load(input_dir / "g_pids.npy")

argsort_distmat = np.argsort(distmat, axis=1)
# print(f"Argsort of distance matrix:\n {argsort_distmat}")


# q_camids = np.array([1, 2, 3, 1, 2, 3, 4, 5])
# g_camids = np.array([1, 1, 2, 2, 3, 3, 4, 4])

pid_matches: NDArray[np.bool_] = g_pids[np.newaxis, :] == q_pids[:, np.newaxis]
# print(f"PID matches:\n {pid_matches.astype(np.int8)}")

# exclude_mask = (g_pids[np.newaxis, :] == q_pids[:, np.newaxis]) & (
#     g_camids[np.newaxis, :] == q_camids[:, np.newaxis]
# )
# include_mask = (g_pids[np.newaxis, :] != q_pids[:, np.newaxis]) | (
#     g_camids[np.newaxis, :] != q_camids[:, np.newaxis]
# )

# ======

# Get sorted pid_matches based on the argsort of the distance matrix
sorted_pid_matches: NDArray[np.bool_] = pid_matches[
    np.arange(pid_matches.shape[0])[:, np.newaxis], argsort_distmat
]
# print(f"Sorted PID matches:\n {sorted_pid_matches.astype(np.int8)}")

# ======

cmc = get_cmc_curve(sorted_pid_matches)
print(f"CMC curve:\n {cmc[:10]}")  # Print first 10 ranks

# ======

# print(f"Average Precision (AP) for each query:\n {average_precision}")

# Compute mean Average Precision (mAP)
average_precision = get_average_precision(sorted_pid_matches)
mean_average_precision = np.mean(average_precision)
print(f"Mean Average Precision (mAP): {mean_average_precision:.1%}")
for r in [1, 5, 10]:
    print(f"CMC curve, Rank-{r:<3}: {cmc[r - 1]:.1%}")