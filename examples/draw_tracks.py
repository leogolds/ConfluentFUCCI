from pathlib import Path

import napari
import numpy as np
import pandas as pd
import vispy
from dask_image.imread import imread
from tifffile import imwrite, imsave, tifffile

from confluentfucci.data import fetch_trackmate_settings
from confluentfucci.gui import CollectiveStats
from confluentfucci.utils import read_stack, run_trackmate
from confluentfucci.math import (
    TrackmateXML,
    CartesianSimilarityFromFile,
    CartesianSimilarity,
)


# data_dir_path = r"C:\Users\leo\AppData\Local\confluentfucci\confluentfucci\Cache\v0.0.19\data\60_frames"
# data_dir_path = r"E:\trying cFUCCI\7.7.22 - ser5\quarter image"
# data_dir_path = r"E:\trying cFUCCI\7.7.22 - ser5"
# data_dir_path = r"E:\trying cFUCCI\tested"
# data_dir_path = r"D:\Data\full_pipeline_tests\left_60_frames"
# data_dir_path = r"C:\Users\leo\PycharmProjects\MicroscopyPipeline\experiments\our_data_on_fuccitrack"
# data_dir_path = r"E:\trying cFUCCI\Areej\8.08 experiment\P19_160-277"
# data_dir_path = r"E:\trying cFUCCI\Areej\8.08 experiment\P20_159-277"
data_dir_path = r"E:\trying cFUCCI\Areej\8.08 experiment\P21_165-277"

# run_trackmate(fetch_trackmate_settings(), data_path=Path(data_dir_path) / 'red_segmented.tiff')
# run_trackmate(fetch_trackmate_settings(), data_path=Path(data_dir_path) / 'green_segmented.tiff')

tm_red = TrackmateXML(Path(data_dir_path) / "red_segmented.tiff.xml")
tm_green = TrackmateXML(Path(data_dir_path) / "green_segmented.tiff.xml")

shape = read_stack(Path(data_dir_path) / "red.tif").shape[1:]

metric_path = Path(data_dir_path) / "metric.h5"
if metric_path.exists():
    metric_df = pd.read_hdf(metric_path, key="metric")
    metric = CartesianSimilarityFromFile(tm_red, tm_green, metric_df, shape=shape)
else:
    metric = CartesianSimilarity(tm_red, tm_green, shape=shape)
    metric.calculate_metric_for_all_tracks()
    metric.metric_df.to_hdf(metric_path, key="metric")

red_df, green_df, yellow_df = metric.get_tracks_for_visualization()
red_df["constant"] = 1
green_df["constant"] = 128
yellow_df["constant"] = yellow_df.color.map({"red": 0, "green": 1, "yellow": 0.5})


# red_df = red_df.query('TrackID in [72, 77, 76]')
red_tracks = list(
    red_df[["TrackID", "frame", "POSITION_Y", "POSITION_X"]].itertuples(
        index=False, name=None
    )
)
green_tracks = list(
    green_df[["TrackID", "frame", "POSITION_Y", "POSITION_X"]].itertuples(
        index=False, name=None
    )
)
yellow_tracks = list(
    yellow_df[["TrackID", "frame", "POSITION_Y", "POSITION_X"]].itertuples(
        index=False, name=None
    )
)

red = imread(Path(data_dir_path) / "red.tif")
green = imread(Path(data_dir_path) / "green.tif")
phase = imread(Path(data_dir_path) / "phase.tif")

viewer = napari.Viewer()
viewer.add_image(
    phase,
    multiscale=False,
    blending="additive",
    opacity=1,
    contrast_limits=[phase.min().compute(), phase.max().compute()],
)
viewer.add_image(
    red,
    multiscale=False,
    blending="additive",
    opacity=1,
    contrast_limits=[red.min().compute(), red.max().compute()],
)
viewer.add_image(
    green,
    multiscale=False,
    blending="additive",
    opacity=1,
    contrast_limits=[green.min().compute(), green.max().compute()],
)

red_cmap = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
green_cmap = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# yellow_cmap = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
yellow_cmap = vispy.color.Colormap(
    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    interpolation="zero",
)

tail = {'tail_length': 30, 'tail_width': 4}

viewer.add_tracks(
    red_tracks,
    name="Red Tracks",
    features=red_df,
    color_by="constant",
    blending="translucent",
    # colormap="hsv",
    colormaps_dict={"constant": red_cmap},
    **tail,
)
viewer.add_tracks(
    green_tracks,
    name="Green Tracks",
    features=green_df,
    color_by="constant",
    blending="translucent",
    # colormap="hsv",
    colormaps_dict={"constant": green_cmap},
    **tail,

)
viewer.add_tracks(
    yellow_tracks,
    name="Yellow Tracks",
    features=yellow_df,
    color_by="constant",
    # color_by="color",
    blending="translucent",
    # colormap="hsv",
    # colormaps_dict={
    #     "constant": yellow_df.color.map(
    #         {"red": red_cmap, "green": green_cmap, "yellow": yellow_cmap}
    #     ).values
    # },
    colormaps_dict={"constant": yellow_cmap},
    **tail,

)

# napari.run()

viewer.dims.set_current_step(0, 0)
np_stack = np.zeros(shape=[*red.shape, 4])
for frame in range(red.shape[0]):
    scrnsht = viewer.screenshot(size=red.shape[1:])
    np_stack[frame, ...] = scrnsht
    viewer.dims.set_current_step(0, frame+1)

    print()
    # break
tifffile.imwrite(
            # "tracks_cellpose.tif",
            Path(data_dir_path) / "tracks_video.tif",
            np_stack,
            metadata={"axes": "ZYXC"},
        )