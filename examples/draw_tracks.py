from pathlib import Path

import napari
import numpy as np
import pandas as pd
import vispy
from dask_image.imread import imread

from confluentfucci.gui import CollectiveStats
from confluentfucci.utils import read_stack
from confluentfucci.math import (
    TrackmateXML,
    CartesianSimilarityFromFile,
    CartesianSimilarity,
)


data_dir_path = r"C:\Users\leo\AppData\Local\confluentfucci\confluentfucci\Cache\v0.0.19\data\60_frames"
# data_dir_path = r"E:\trying cFUCCI\7.7.22 - ser5\quarter image"
# data_dir_path = r"E:\trying cFUCCI\7.7.22 - ser5"

tm_red = TrackmateXML(Path(data_dir_path) / "red_segmented.tiff.xml")
tm_green = TrackmateXML(Path(data_dir_path) / "green_segmented.tiff.xml")

shape = read_stack(Path(data_dir_path) / "red.tif").shape[1:]

metric_path = Path(data_dir_path) / "metric.h5"
if metric_path.exists():
    metric_df = pd.read_hdf(metric_path, key="metric")
    metric = CartesianSimilarityFromFile(tm_red, tm_green, metric_df, shape=shape)
else:
    metric = CartesianSimilarity(tm_red, tm_green, shape=shape)

red_df, green_df, yellow_df = metric.get_tracks_for_visualization()
red_df["constant"] = 1
green_df["constant"] = 128
yellow_df["constant"] = 254

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

viewer = napari.Viewer()
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
yellow_cmap = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

viewer.add_tracks(
    red_tracks,
    name="Red Tracks",
    features=red_df,
    color_by="constant",
    blending='translucent',
    # colormap="hsv",
    colormaps_dict={'constant': red_cmap},
)
viewer.add_tracks(
    green_tracks,
    name="Green Tracks",
    features=green_df,
    color_by="constant",
    blending='translucent',
    # colormap="hsv",
    colormaps_dict={'constant': green_cmap},

)
viewer.add_tracks(
    yellow_tracks,
    name="Yellow Tracks",
    features=yellow_df,
    color_by="constant",
    blending='translucent',
    # colormap="hsv",
    colormaps_dict={'constant': yellow_cmap},

)

napari.run()
# spots_df = metric.get_all_spots()
# fucci_tracks = spots_df.query('merged_track_id != "unmerged"').copy()
# individual_tracks = spots_df.query('merged_track_id == "unmerged"').copy()
# fucci_tracks["numerical_track_id"], _ = pd.factorize(fucci_tracks.merged_track_id)
#
# tracks = (
#     fucci_tracks[["numerical_track_id", "frame", "POSITION_Y", "POSITION_X"]]
#     .sort_values(["numerical_track_id", "frame"])
#     .values
# )
# a = (
#     fucci_tracks[["numerical_track_id", "frame", "POSITION_Y", "POSITION_X"]]
#     .sort_values(["numerical_track_id", "frame"])
#     .astype(
#         {
#             "numerical_track_id": int,
#             "frame": int,
#             "POSITION_Y": float,
#             "POSITION_X": float,
#         }
#     )
# )
# tracks = [
#     (r.numerical_track_id, r.frame, r.POSITION_Y, r.POSITION_X) for i, r in a.iterrows()
# ]
