"""Basic utils for running the analysis."""

from __future__ import annotations

import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil
import tifffile

# from aicsimageio.transforms import reshape_data
from aicsimageio.writers import OmeTiffWriter
from cellpose import models
from docker.client import DockerClient
from docker.types import Mount
from scipy.spatial import Voronoi
from tqdm import trange


def read_stack(path: Path) -> np.ndarray:
    """Read stacks saved as an h5 or tiff
    For h5 data should be in either data/exported_data.
    """
    if path.suffix == "h5":
        f = h5py.File(path)
        return f.get("data", f.get("exported_data"))
    else:
        return tifffile.imread(path)


def segment_stack(
    path: Path,
    model: Path,
    export_tiff: bool = True,
    panel_red_tqdm_instance=None,
    panel_green_tqdm_instance=None,
):
    """Segment stack frame by frame."""
    print(f"segmenting stack at {path} with model at {model}")
    stack = read_stack(path)

    frames, Y, X = stack.shape

    new_file_path = path.parent / f"{path.stem}_segmented.h5"
    dataset_name = "data"

    with h5py.File(new_file_path, "w") as f:
        dataset = f.create_dataset(
            dataset_name,
            shape=(frames, Y, X),
            dtype=np.uint16,
            chunks=True,
        )

        if panel_red_tqdm_instance:
            frame_iterator = panel_red_tqdm_instance(
                range(frames), desc="Red", colour="#ff0000"
            )
        elif panel_green_tqdm_instance:
            frame_iterator = panel_green_tqdm_instance(
                range(frames), desc="Green", colour="#008000"
            )
        else:
            frame_iterator = trange(frames)

        for frame in frame_iterator:
            masks, center_of_mass = _segment_frame(stack[frame, ...], model, gpu=True)
            dataset[frame, :, :] = masks

        if export_tiff:
            new_tiff_path = path.parent / f"{path.stem}_segmented.tiff"
            print(f"exporting to tiff at {new_tiff_path}")
            # TODO wtf? why is tifffile stopped writing TrackMate compatible files?
            OmeTiffWriter.save(
                f.get(dataset_name).__array__(), new_tiff_path, dim_order="TYX"
            )
            # reshaped = reshape_data(f.get(dataset_name), "TYX", "ZTCYX")
            # tifffile.imwrite(new_tiff_path, f.get(dataset_name), bigtiff=True)
            # with tifffile.TiffWriter(new_tiff_path, bigtiff=True) as tif:
            #     tif.write(f.get(dataset_name), shape=(frames, Y, X), metadata={'axes': 'TYX'})
            # tif.write(reshaped, shape=reshaped.shape)

    print("segmentation complete")


def _segment_frame(img, model: Path, gpu: bool = False, diameter: int = 18):
    channels = [0, 0]
    net_avg = False
    resample = False

    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=str(model.absolute()),
        nchan=2,
    )
    masks, flows, styles = model.eval(
        img.astype(np.float16, copy=False),
        diameter=diameter,
        channels=channels,
        net_avg=net_avg,
        resample=resample,
        z_axis=0,
    )

    return masks, None


def run_trackmate(settings_path: Path, data_path: Path) -> None:
    print("TOP LEVEL run_trackmate")
    if os.environ.get("DOCKER"):
        print("run _run_local_trackmate")

        return _run_local_trackmate(settings_path, data_path)
    print("run _run_trackmate")
    _run_trackmate(settings_path, data_path)


def _run_local_trackmate(settings_path: Path, data_path: Path):
    print("RUN LOCAL DOCKER")
    # cmd = f"/opt/fiji/ImageJ-linux64 --ij2 --headless --console --memory=$MEMORY --run read_settings_and_process_tiff_stack.py"
    cmd = [
        "/opt/fiji/Fiji.app/ImageJ-linux64",
        "--ij2",
        "--headless",
        "--console",
        f"--memory={int(psutil.virtual_memory().total // 1024 ** 3 * 0.5)}G",
        "--run",
        "/workspace/read_settings_and_process_tiff_stack.py",
    ]

    env = {
        **os.environ,
        "DOCKER_SETTINGS_XML": str(settings_path.absolute()),
        "DOCKER_TIFF_STACK": str(data_path.absolute()),
        # "MEMORY": f"{int(psutil.virtual_memory().total // 1024 ** 3 * 0.5)}G",
    }
    subprocess.run(cmd, env=env, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)


def _run_trackmate(settings_path: Path, data_path: Path) -> None:
    """Run TrackMate through a custom container using DockerClient."""
    print(
        f"Running TrackMate on segmented stack at {data_path} using settings at {settings_path}",
    )
    settings_mount = Mount(
        target="/settings",
        source=str(settings_path.parent),
        # source=str(settings_path.parent.absolute()),
        type="bind",
        read_only=True,
    )
    data_mount = Mount(
        target="/data",
        source=str(data_path.parent),
        # source=str(data_path.parent.absolute()),
        type="bind",
        read_only=False,
    )

    container = _docker_client.containers.run(
        image="leogold/trackmate:v1",
        detach=True,
        mounts=[settings_mount, data_mount],
        environment={
            "SETTINGS_XML": settings_path.name,
            "TIFF_STACK": data_path.name,
            "MEMORY": f"{int(psutil.virtual_memory().total // 1024**3 * 0.5)}G",
        },
    )

    for line in container.logs(stream=True):
        print(line.decode("utf-8"))

    print(f"Tracking on {data_path} complete")


_docker_client = None


def get_docker_client():
    global _docker_client
    try:
        if _docker_client:
            return _docker_client
        else:
            _docker_client = DockerClient()
            return _docker_client
    except Exception as e:
        print(traceback.format_exc())
        return None


get_docker_client()
# try:
#     _docker_client = DockerClient()
# except Exception:
#     print("Failed to initialize local Docker client, running TrackMate disabled")
#     _docker_client = None
