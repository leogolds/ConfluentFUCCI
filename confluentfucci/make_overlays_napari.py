from pathlib import Path
import itertools

import napari
from skimage.exposure import equalize_adapthist
from skimage.io import imsave

from confluentfucci import utils

base_data_path = Path(r"D:\Data\full_pipeline_tests\left_60_frames")
phase = utils.read_stack(base_data_path / "phase.tif")
red_seg = utils.read_stack(base_data_path / "red_segmented.tiff")
green_seg = utils.read_stack(base_data_path / "green_segmented.tiff")


def create_overlay(frame=0):
    # add the image
    microns_per_pixel = [0.67, 0.67]
    viewer = napari.view_image(
        equalize_adapthist(phase[frame, ...]),
        scale=microns_per_pixel,
        blending="additive",
    )
    viewer.add_image(
        red_seg[frame, ...] != 0,
        colormap="red",
        scale=microns_per_pixel,
        opacity=1,
        blending="additive",
    )
    viewer.add_image(
        green_seg[frame, ...] != 0,
        colormap="green",
        scale=microns_per_pixel,
        opacity=1,
        blending="additive",
    )
    # viewer.scale_bar.visible = True
    # viewer.scale_bar.unit = "um"
    # viewer.camera.zoom = 2

    # viewer
    imsave(
        f"cellpose_overlay/cellpose_overlay_{frame:03}.png",
        viewer.screenshot(size=phase.shape[1:],scale=2),
    )


[create_overlay(f) for f in range(45,50)]
