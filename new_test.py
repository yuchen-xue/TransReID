from argparse import ArgumentParser
from collections.abc import Container
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode

from config import cfg
from model import make_model

SIZE_TEST = (256, 128)
PIXEL_MEAN = (0.5, 0.5, 0.5)
PIXEL_STD = (0.5, 0.5, 0.5)
IMS_PER_BATCH = 256

VIEW_LABEL_VALUE = 1
DEVICE = "cuda"
NUM_WORKERS = cpu_count()


class UniversalDataset[T](Dataset):
    def __init__(self, value: Container[T]):
        self.value = value

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, index: int) -> T:
        return self.value[index]


def get_pid_from_path(img_path: Path) -> int:
    return int(img_path.name.split("_")[0])


def get_camid_from_path(img_path: Path) -> int:
    return int(img_path.name.split("_")[1][1]) - 1


def imread_collate(img_paths: Container[str | Path]) -> torch.Tensor:
    imgs = (Image.open(img_path).convert("RGB") for img_path in img_paths)
    transforms = T.Compose(
        (T.Resize(SIZE_TEST), T.ToTensor(), T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD))
    )
    batch_imgs = torch.stack(tuple(transforms(img) for img in imgs), dim=0)

    return batch_imgs


def get_camid_collate(img_paths: Container[str | Path]) -> torch.Tensor:
    camids = tuple(get_camid_from_path(img_path) for img_path in img_paths)
    return torch.tensor(camids, dtype=torch.int8)


def euclidean_distance(
    qf: NDArray[np.float64], gf: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Args:
        qf: query features, shape [m, d]
        gf: gallery features, shape [n, d]

    Returns:
        distmat: distance matrix, shape [m, n]
    """
    qf_squared = np.square(qf).sum(axis=1, keepdims=True)
    gf_squared = np.square(gf).sum(axis=1, keepdims=True).T
    return qf_squared + gf_squared - 2 * qf @ gf.T


def run_inference(cfg, query_img_paths, gallery_img_paths):
    print("Running inference using the new run_inference function...")

    # Get number of classes and number of cameras from the training dataset
    train_img_root = Path(cfg.DATASETS.ROOT_DIR) / "market1501/bounding_box_train"
    num_classes = len(
        {get_pid_from_path(img_path) for img_path in train_img_root.glob("*.jpg")}
    )
    camera_num = len(
        {get_camid_from_path(img_path) for img_path in train_img_root.glob("*.jpg")}
    )
    gallery_img_dataset = UniversalDataset[Path](gallery_img_paths)
    gallery_img_dataloader = DataLoader(
        gallery_img_dataset,
        batch_size=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
        collate_fn=imread_collate,
    )
    gallery_camid_dataloader = DataLoader(
        gallery_img_dataset,
        batch_size=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
        collate_fn=get_camid_collate,
    )
    gallery_view_label_loader = DataLoader(
        UniversalDataset[int](
            (torch.tensor(VIEW_LABEL_VALUE, dtype=torch.int8),) * len(gallery_img_paths)
        ),
        batch_size=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
    )

    # Prepare query dataloader
    query_img_dataset = UniversalDataset[Path](query_img_paths)
    query_img_dataloader = DataLoader(
        query_img_dataset,
        batch_size=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
        collate_fn=imread_collate,
    )

    query_camid_dataloader = DataLoader(
        query_img_dataset,
        batch_size=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
        collate_fn=get_camid_collate,
    )

    # Create a view_label that has the same shape as each batch of images, with all values set to 1
    # (since we don't have view labels in this case)
    query_view_label_loader = DataLoader(
        UniversalDataset[int](
            (torch.tensor(VIEW_LABEL_VALUE, dtype=torch.int8),) * len(query_img_paths)
        ),
        batch_size=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
    )

    # Load model and extract features
    print(
        f"Creating model with num_classes={num_classes}, camera_num={camera_num}, view_num=1"
    )
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=1)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        query_feats_tuple = tuple(
            model(
                img.to(DEVICE),
                cam_label=cam_label.to(DEVICE),
                view_label=view_label.to(DEVICE),
            )
            .detach()
            .cpu()
            for img, cam_label, view_label in zip(
                query_img_dataloader,
                query_camid_dataloader,
                query_view_label_loader,
            )
        )
        gallery_feats_tuple = tuple(
            model(
                img.to(DEVICE),
                cam_label=cam_label.to(DEVICE),
                view_label=view_label.to(DEVICE),
            )
            .detach()
            .cpu()
            for img, cam_label, view_label in zip(
                gallery_img_dataloader,
                gallery_camid_dataloader,
                gallery_view_label_loader,
            )
        )

    query_feats = torch.cat(query_feats_tuple)
    gallery_feats = torch.cat(gallery_feats_tuple)

    # Normalize features using L2 normalization
    query_feats = torch.nn.functional.normalize(query_feats).numpy()
    gallery_feats = torch.nn.functional.normalize(gallery_feats).numpy()

    return query_feats, gallery_feats


def get_cmc_curve(matches: NDArray[np.bool_]) -> NDArray[np.float64]:

    # Compute top-k accuracy for each query
    cumsum_matches = matches.cumsum(axis=1)
    cumsum_matches[cumsum_matches > 1] = 1
    top_k_acc = cumsum_matches

    # Compute CMC (Cumulative Matching Characteristic) curve
    cmc = top_k_acc.mean(axis=0)
    return cmc


def get_average_precision(matches: NDArray[np.bool_]) -> NDArray[np.float64]:

    # Compute top-k precision for each query
    cumsum_matches = matches.cumsum(axis=1)
    rank_position = np.arange(1, cumsum_matches.shape[1] + 1)
    top_k_precision = cumsum_matches / rank_position
    # print(f"{top_k_precision.shape=}, {top_k_precision.dtype=}")
    # print(f"Overview of Top-k Precision:\n {top_k_precision[:5, :10]}")  # Print first 5 queries and first 10 ranks

    # Compute Average Precision (AP) for each query
    matched_cum_prec = top_k_precision * matches
    num_rel = matches.sum(axis=1)
    average_precision = matched_cum_prec.sum(axis=1) / num_rel
    return average_precision


def get_cmc_mAP(
    dist_mat: NDArray[np.float64],
    query_pid_arr: NDArray[np.int_],
    gallery_pid_arr: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate the CMC curve and mean Average Precision (mAP) given a distance matrix and corresponding PIDs.

    Args:
        dist_mat: Distance matrix of shape [num_queries, num_gallery]
        query_pid_arr: Array of query PIDs of shape [num_queries]
        gallery_pid_arr: Array of gallery PIDs of shape [num_gallery]
    """

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
    average_precision = get_average_precision(sorted_pid_matches)

    return cmc, average_precision


def main(cfg: CfgNode) -> None:

    query_img_root = Path(cfg.DATASETS.ROOT_DIR) / "market1501/query"
    gallery_img_root = Path(cfg.DATASETS.ROOT_DIR) / "market1501/bounding_box_test"

    query_img_paths = tuple(
        img_path
        for img_path in sorted(query_img_root.glob("*.jpg"))
        if get_pid_from_path(img_path) != -1
    )

    # Prepare gallery dataloader
    gallery_img_paths = tuple(
        img_path
        for img_path in sorted(gallery_img_root.glob("*.jpg"))
        if get_pid_from_path(img_path) != -1
    )

    query_feats, gallery_feats = run_inference(cfg, query_img_paths, gallery_img_paths)

    query_pid_arr = np.array(
        tuple(get_pid_from_path(img_path) for img_path in query_img_paths)
    )
    gallery_pid_arr = np.array(
        tuple(get_pid_from_path(img_path) for img_path in gallery_img_paths)
    )

    dist_mat = euclidean_distance(query_feats, gallery_feats)

    cmc_curve, average_precision = get_cmc_mAP(dist_mat, query_pid_arr, gallery_pid_arr)

    mean_average_precision = np.mean(average_precision)
    print(f"Mean Average Precision (mAP): {mean_average_precision:.1%}")
    for r in (1, 5, 10):
        print(f"CMC curve, Rank-{r:<3}: {cmc_curve[r - 1]:.1%}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to the config file")
    args = parser.parse_args()

    cfg = CfgNode()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    main(cfg)
