import itertools
import xml.etree.ElementTree as et
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial import Voronoi
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.geometry.linestring import LineString

tqdm.tqdm.pandas()

# TODO externalize
magnification_towards_camera = 1
# pixel_size_in_microns = 0.345 * magnification_towards_camera
pixel_size_in_microns = 0.67 * magnification_towards_camera
calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2


def pairwise_iterator(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def make_polygon(df):
    polygon = Polygon(df.ROI)
    x, y = df.POSITION_X, df.POSITION_Y
    polygon = translate(polygon, x + 0.5, y + 0.5)

    return polygon


spots_relevant_columns = [
    "frame",
    "POSITION_X",
    "POSITION_Y",
    "PERIMETER",
    "image_id",
    "AREA",
    "ROI",
    "roi_polygon",
    "ELLIPSE_MAJOR",
    "ELLIPSE_MINOR",
]
tracks_relevant_columns = [
    "EDGE_TIME",
    "TrackID",
    "SPOT_SOURCE_ID",
    "SPOT_TARGET_ID",
    "EDGE_X_LOCATION",
    "EDGE_Y_LOCATION",
]


class TrackmateXML:
    """
    Derived from https://github.com/rharkes/pyTrackMateXML/blob/master/trackmatexml.py and updated with custom features
    Trackmate-xml is a class around trackmate xml files to simplify some typical operations on the files, while still
    maintaining access to the raw data.
    """

    class_version = 1.0

    def __init__(self, filename):
        if isinstance(filename, str):
            self.pth = Path(filename)
        elif isinstance(filename, Path):
            self.pth = filename
        else:
            raise ValueError("not a valid filename")

        if self.pth.suffix == ".h5":
            store = pd.HDFStore(self.pth)
            self.spots = store.spots
            self.tracks = store.tracks
            self.filteredtracks = store.filtered_tracks
            other_info = store.other_info
            self.version = other_info.version[0]
            store.close()
        elif self.pth.suffix == ".xml":
            etree = et.parse(self.pth)
            root = etree.getroot()
            if not root.tag == "TrackMate":
                raise ValueError("Not a TrackmateXML")
            self.version = root.attrib["version"]
            self.spots = self.__loadspots(root)
            self.tracks = self.__loadtracks(root)
            self.filteredtracks = self.__loadfilteredtracks(root)
        else:
            raise ValueError("{0} is not avalid file suffix".format(self.pth.suffix))

    def save(self, filename, create_new=True):
        """
        Saves the spots, tracks and filteredtracks to an HDFStore
        """
        if isinstance(filename, str):
            pth = Path(filename)
        elif isinstance(filename, Path):
            pth = filename
        else:
            raise ValueError("not a valid filename")
        if pth.exists() & create_new:
            pth.unlink()
        store = pd.HDFStore(pth)
        store["spots"] = self.spots
        store["tracks"] = self.tracks
        store["filtered_tracks"] = self.filteredtracks
        other_info = {
            "version": self.version,
            "class_version": TrackmateXML.class_version,
        }
        store["other_info"] = pd.DataFrame(other_info, index=[0])
        store.close()

    @staticmethod
    def __loadfilteredtracks(root):
        """
        Loads all filtered tracks from xml
        :param root: root of xml
        :return: filtered tracks
        """
        filtered_tracks = []
        for track in root.iter("TrackID"):
            track_values = track.attrib
            track_values["TRACK_ID"] = int(track_values.pop("TRACK_ID"))
            filtered_tracks.append(track_values)
        ftracks = pd.DataFrame(filtered_tracks)
        return ftracks

    @staticmethod
    def __loadtracks(root):
        """
        load all tracks in the .xml file
        :param root: root of .xml file
        :return: tracks as pandas dataframe
        """
        all_tracks = []
        for track in root.iter("Track"):
            curr_track = int(track.attrib["TRACK_ID"])
            all_edges = []
            for edge in track:
                edge_values = edge.attrib
                edge_values["SPOT_SOURCE_ID"] = int(edge_values.pop("SPOT_SOURCE_ID"))
                edge_values["SPOT_TARGET_ID"] = int(edge_values.pop("SPOT_TARGET_ID"))
                edge_values["TrackID"] = curr_track
                all_edges.append(edge_values)
            all_tracks.append(pd.DataFrame(all_edges))
        tracks = pd.concat(all_tracks)
        # return tracks
        # TODO align track and spots ID field usage
        return tracks[tracks_relevant_columns]

    @staticmethod
    def __loadspots(root):
        """
        Loads all spots in the xml file
        :return: spots as pandas dataframe
        """
        # load all spots
        all_frames = []
        for spots_in_frame in root.iter("SpotsInFrame"):
            curr_frame = spots_in_frame.attrib["frame"]
            # go over all spots in the frame
            all_spots = []
            for spot in spots_in_frame:
                spot_values = spot.attrib
                spot_values.pop("name")  # not needed
                spot_values["frame"] = int(curr_frame)
                spot_values["ID"] = int(spot_values.pop("ID"))  # we want ID to be integer, so we can index later
                spot_values["POSITION_X"] = float(spot_values.pop("POSITION_X"))
                spot_values["POSITION_Y"] = float(spot_values.pop("POSITION_Y"))
                spot_values["image_id"] = int(float(spot_values.get("MAX_INTENSITY_CH1")))
                spot_values["ROI"] = [(float(x), float(y)) for x, y in pairwise_iterator(spot.text.split(" "))]
                all_spots.append(spot_values)
            all_frames.append(pd.DataFrame(all_spots))

        spots = pd.concat(all_frames)
        spots.set_index("ID", inplace=True, verify_integrity=True)
        # spots = spots.astype("float")

        spots["roi_polygon"] = spots.apply(make_polygon, axis="columns")
        spots["AREA"] = pd.to_numeric(spots.AREA)

        # return spots
        return spots[spots_relevant_columns]

    @cache
    def trace_track(self, track_id, verbose=False):
        """
        Traces a track over all spots.
        :param verbose: report if a split is found
        :param track_id:
        """
        # assert isinstance(track_id, int)
        # Tracks consist of edges. The edges are not sorted
        current_track = self.tracks[self.tracks["TrackID"] == track_id]
        if current_track.empty:
            raise ValueError("track {0} not found".format(track_id))
        track_splits = []
        source_spots = self.spots.loc[current_track["SPOT_SOURCE_ID"].values].reset_index()
        target_spots = self.spots.loc[current_track["SPOT_TARGET_ID"].values].reset_index()
        currentindex = source_spots["frame"].idxmin()
        whole_track = [source_spots.loc[currentindex], target_spots.loc[currentindex]]
        # can we continue from the target to a new source?
        current_id = target_spots["ID"].loc[currentindex]
        currentindex = source_spots.index[source_spots["ID"] == current_id].tolist()
        while len(currentindex) > 0:
            if len(currentindex) > 1:
                currentindex = currentindex[0]
                fr = target_spots["frame"].loc[currentindex]
                if verbose:
                    print("Got a split at frame {0} Will continue on branch 0".format(int(fr)))
                    # but so far we do nothing with this knowledge
                track_splits.append(fr)
            else:
                currentindex = currentindex[0]
            whole_track.append(target_spots.loc[currentindex])
            current_id = target_spots["ID"].loc[currentindex]
            currentindex = source_spots.index[source_spots["ID"] == current_id].tolist()
        whole_track = pd.concat(whole_track, axis=1).T.reset_index(drop=True)

        # line = LineString(
        #     whole_track[["POSITION_X", "POSITION_Y"]]
        #     .astype(float)
        #     .itertuples(index=False, name=None)
        # )
        whole_track["track_id"] = str(int(track_id))
        return whole_track  # , track_splits
        # return line, whole_track, track_splits


def filter_voronoi_tiling(df, rect, quantile=0.99):
    in_bounds_tiles_df = df.loc[df.apply(lambda row: rect.contains(Polygon(row.vertices)), axis="columns")]
    return in_bounds_tiles_df.query("area < area.quantile(@quantile)")


def compute_voronoi(df):
    # print(df.head())
    # vor = Voronoi(df[["center_x", "center_y"]])
    vor = Voronoi(df[["POSITION_X", "POSITION_Y"]])
    # = pd.DataFrame()

    df["vertice_ids"] = [vor.regions[i] for i in vor.point_region]
    df["valid_region"] = [True if min(l) != -1 else False for l in df.vertice_ids]
    df["vertices"] = [
        np.array([vor.vertices[vertice_id] for vertice_id in vertice_ids]) for vertice_ids in df.vertice_ids
    ]

    # global global_df_list
    # global_df = pd.concat([global_df, df])
    # global_df_list.append(df)
    return df


def compute_voronoi_stats(df):
    df["area"] = [Polygon(vert).area * calibration_squared_microns_to_squared_pixel for vert in df.vertices]
    df["perimeter"] = [Polygon(vert).length * pixel_size_in_microns for vert in df.vertices]

    # horizontal_bins = range(0, 4096, 40)
    # df["bins"] = pd.cut(
    #     df.center_x, bins=horizontal_bins, labels=range(len(horizontal_bins) - 1)
    # )

    # global_stats_list.append(df)
    return df


class CartesianSimilarity:
    def __init__(self, tm_red: TrackmateXML, tm_green: TrackmateXML):
        self.metric_df = pd.DataFrame()
        self.tm_red = tm_red
        self.tm_green = tm_green

    @cache
    def calculate_metric(self, green_track_id, red_track_id):
        red_track_df = self.tm_red.trace_track(red_track_id)
        green_track_df = self.tm_green.trace_track(green_track_id)
        min_frame = max(red_track_df.frame.min(), green_track_df.frame.min())
        max_frame = min(red_track_df.frame.max(), green_track_df.frame.max())

        red_track_df = red_track_df.query("@min_frame <= frame <= @max_frame")
        green_track_df = green_track_df.query("@min_frame <= frame <= @max_frame")

        if len(red_track_df) < 5 or len(green_track_df) < 5:
            return np.inf

        sse = (
            (
                (red_track_df.reset_index().POSITION_X - green_track_df.reset_index().POSITION_X) ** 2
                + (red_track_df.reset_index().POSITION_Y - green_track_df.reset_index().POSITION_Y) ** 2
            )
            ** 0.5
        ).sum()

        return sse / (max_frame - min_frame)

    def calculate_metric_for_all_tracks(self):
        red_track_ids = self.tm_green.tracks.TrackID.unique().tolist()
        green_track_ids = self.tm_green.tracks.TrackID.unique().tolist()
        combinations = list(
            itertools.product(
                red_track_ids,
                green_track_ids,
            )
        )

        metrics = [
            self.calculate_metric(g, r) for r, g in tqdm.tqdm(combinations, desc="Calculating similarity metric")
        ]
        df = pd.DataFrame(columns=["red_track", "green_track"], data=combinations)
        df["metric"] = metrics

        self.metric_df = df.sort_values("metric").reset_index(drop=True)
        return self.metric_df

    @cache
    def merge_tracks(self, red_track_id, green_track_id):
        red_track_df = self.tm_red.trace_track(red_track_id)
        green_track_df = self.tm_green.trace_track(green_track_id)

        overlap_frame_min = max(red_track_df.frame.min(), green_track_df.frame.min())
        overlap_frame_max = min(red_track_df.frame.max(), green_track_df.frame.max())

        overlap_red_frames = red_track_df.query("@overlap_frame_min <= frame <= @overlap_frame_max+1")
        overlap_green_frames = green_track_df.query("@overlap_frame_min <= frame <= @overlap_frame_max+1")
        # df = red_track_df.merge(
        #     green_track_df,
        #     on="frame",
        #     how="outer",
        #     suffixes=("_red", "_green"),
        #     indicator=True,
        # )

        rows = []
        for frame in range(overlap_frame_min, overlap_frame_max + 1):
            r = red_track_df.query("frame == @frame")
            g = green_track_df.query("frame == @frame")

            row = (r if g.empty else g).copy()
            row["source_track"] = "red" if g.empty else "green"
            if not r.empty and not g.empty:
                row = (r if r.AREA.values > g.AREA.values else g).copy()
                row["source_track"] = "red" if r.AREA.values > g.AREA.values else "green"
                row["POSITION_Y"] = np.mean([r.POSITION_X, g.POSITION_X])
                row["POSITION_Y"] = np.mean([r.POSITION_Y, g.POSITION_Y])
            rows.append(row)
        yellow_frames = pd.concat(rows) if rows else pd.DataFrame(columns=["source_track", *red_track_df.columns])

        # yellow_frames = (
        #     pd.concat([overlap_red_frames, overlap_green_frames])
        #     .groupby("frame")[["frame", "POSITION_X", "POSITION_Y"]]
        #     .mean()
        #     .astype({"frame": "int"})
        # )

        red_frames = red_track_df.query("frame < @overlap_frame_min or frame > @overlap_frame_max").copy()
        green_frames = green_track_df.query("frame < @overlap_frame_min or frame > @overlap_frame_max").copy()

        yellow_frames["color"] = "yellow"
        red_frames["color"] = "red"
        red_frames["source_track"] = "red"
        green_frames["color"] = "green"
        green_frames["source_track"] = "green"

        df = pd.concat([red_frames, green_frames, yellow_frames]).reset_index(drop=True).sort_values("frame")
        df["merged_track_id"] = f"r{int(red_track_id)}_g{int(green_track_id)}"
        return df

    def get_merged_tracks(self, max_metric_value: float = 2.0):
        if self.metric_df.empty:
            self.calculate_metric_for_all_tracks()

        print("Merging tracks")
        track_df_list = (
            self.metric_df.query("metric < @max_metric_value")
            .progress_apply(lambda x: self.merge_tracks(x.red_track, x.green_track), axis="columns")
            .to_list()
        )
        return pd.concat(track_df_list).reset_index(drop=True)

    def count_cells_in_bins(self, bin_labels=("left", "middle", "right")):
        all_cells_df = self.partition_cells_into_bins(bin_labels)

        # red_count = distinctly_red_spots.groupby(["frame", "bin"]).size()
        # green_count = distinctly_green_spots.groupby(["frame", "bin"]).size()
        merged_tracks_count = all_cells_df.groupby(["frame", "color", "bin"]).size().unstack("color")

        df = merged_tracks_count.copy().fillna(0).reset_index()
        # df["green"] = df["green"] + green_count
        # df["red"] = df["red"] + red_count

        return df

    def partition_cells_into_bins(self, bin_labels=("left", "middle", "right")):
        all_spots = self.get_all_spots()

        # red_spots_in_merged_tracks = all_merged_tracks.query('source_track == "red"').ID
        # green_spots_in_merged_tracks = all_merged_tracks.query(
        #     'source_track == "green"'
        # ).ID
        all_spots["bin"] = pd.cut(all_spots.POSITION_X, len(bin_labels), labels=bin_labels)

        return all_spots

    def get_all_spots(self):
        (
            green_spots_in_merged_tracks,
            red_spots_in_merged_tracks,
        ) = self.get_merged_red_green_spot_ids()

        distinctly_red_spots = self.tm_red.spots.drop(red_spots_in_merged_tracks).reset_index()
        distinctly_green_spots = self.tm_green.spots.drop(green_spots_in_merged_tracks).reset_index()

        distinctly_red_spots[["color", "source_track"]] = "red"
        distinctly_red_spots["source_track"] = "red"
        distinctly_green_spots[["color", "source_track"]] = "green"
        distinctly_green_spots["source_track"] = "green"
        distinctly_red_spots["merged_track_id"] = "unmerged"
        distinctly_green_spots["merged_track_id"] = "unmerged"

        # distinctly_red_spots["track_uid"] = distinctly_red_spots.track_id.apply(
        #     lambda x: f"r{x:d}"
        # )
        # distinctly_green_spots["track_uid"] = distinctly_green_spots.track_id.apply(
        #     lambda x: f"g{x:d}"
        # )
        merged = self.get_merged_tracks()
        # merged["track_uid"] = merged.merged_track_id
        #
        df = pd.concat([merged, distinctly_red_spots, distinctly_green_spots]).reset_index(drop=True)

        return df

    def get_merged_red_green_spot_ids(self):
        accounted_red_green_track_ids = self.get_track_ids_accounted_by_merge()

        # extract spots accounted for red/green spot ids
        red_spots_in_merged_tracks = pd.concat(
            accounted_red_green_track_ids.red_track_id.apply(self.tm_red.trace_track).values
        ).ID
        green_spots_in_merged_tracks = pd.concat(
            accounted_red_green_track_ids.green_track_id.apply(self.tm_green.trace_track).values
        ).ID

        return green_spots_in_merged_tracks, red_spots_in_merged_tracks

    def get_track_ids_accounted_by_merge(self):
        all_merged_tracks = self.get_merged_tracks()

        # extract track id from merge_id
        accounted_red_green_track_ids = all_merged_tracks.merged_track_id.str.extract(
            r"r(?P<red_track_id>\d*)_g(?P<green_track_id>\d*)"
        )

        accounted_red_green_track_ids["red_track_id"] = pd.to_numeric(accounted_red_green_track_ids.red_track_id)
        accounted_red_green_track_ids["green_track_id"] = pd.to_numeric(accounted_red_green_track_ids.green_track_id)

        return accounted_red_green_track_ids.drop_duplicates()

    def get_unmerged_red_green_tracks(self):
        merged_tracks = self.get_track_ids_accounted_by_merge()

        red_unmerged_tracks = pd.concat(
            merged_tracks.red_track_id.apply(self.tm_red.trace_track).values,
            ignore_index=True,
        )
        red_unmerged_tracks["source_track"] = "red"
        green_unmerged_tracks = pd.concat(
            merged_tracks.green_track_id.apply(self.tm_green.trace_track).values,
            ignore_index=True,
        )
        green_unmerged_tracks["source_track"] = "green"

        return red_unmerged_tracks, green_unmerged_tracks

    def get_binned_unmerged_tracks(self, shape, bin_size=250):
        red_unmerged, green_unmerged = self.get_unmerged_red_green_tracks()
        result_df = pd.concat([red_unmerged, green_unmerged], ignore_index=True)
        x_interval_range = pd.interval_range(start=0, end=shape[1], freq=bin_size)
        result_df["x_grid_interval"] = pd.cut(result_df.POSITION_X, x_interval_range)
        result_df["x_bin"] = result_df.x_grid_interval.cat.rename_categories([int(i.mid) for i in x_interval_range])

        y_interval_range = pd.interval_range(start=0, end=shape[0], freq=bin_size)
        result_df["y_grid_interval"] = pd.cut(result_df.POSITION_Y, y_interval_range)
        result_df["y_bin"] = result_df.y_grid_interval.cat.rename_categories([int(i.mid) for i in y_interval_range])
        #
        result_df["y_bin"] = result_df["y_bin"].astype(np.float32)
        result_df["x_bin"] = result_df["x_bin"].astype(np.float32)

        return result_df

    def get_binned_merged_tracks(self, shape, bin_size=250):
        # merged = metric.get_merged_tracks()
        result_df = self.get_merged_tracks()
        x_interval_range = pd.interval_range(start=0, end=shape[1], freq=bin_size)
        result_df["x_grid_interval"] = pd.cut(result_df.POSITION_X, x_interval_range)
        result_df["x_bin"] = result_df.x_grid_interval.cat.rename_categories([int(i.mid) for i in x_interval_range])

        y_interval_range = pd.interval_range(start=0, end=shape[0], freq=bin_size)
        result_df["y_grid_interval"] = pd.cut(result_df.POSITION_Y, y_interval_range)
        result_df["y_bin"] = result_df.y_grid_interval.cat.rename_categories([int(i.mid) for i in y_interval_range])
        #
        result_df["y_bin"] = result_df["y_bin"].astype(np.float32)
        result_df["x_bin"] = result_df["x_bin"].astype(np.float32)

        return result_df

    def calculate_flow_field(self, shape):
        merged_df = self.get_binned_merged_tracks(shape=shape)
        unmerged_df = self.get_binned_unmerged_tracks(shape=shape)

        # Calculate flow field per track
        print("Calculating flow field")
        merged_df[["d_frame", "d_x", "d_y", "magnitude", "angle"]] = merged_df.groupby(
            "merged_track_id", group_keys=False
        ).progress_apply(calculate_positional_derivative)
        unmerged_df[["d_frame", "d_x", "d_y", "magnitude", "angle"]] = unmerged_df.groupby(
            ["source_track", "track_id"], group_keys=False
        ).progress_apply(calculate_positional_derivative)

        c = (
            pd.concat([merged_df, unmerged_df], ignore_index=True)
            .groupby(["frame", "x_bin", "y_bin"])
            .agg({"angle": "mean", "magnitude": "mean"})
            .reset_index()
        )

        return c


class CartesianSimilarityFromFile(CartesianSimilarity):
    def __init__(self, tm_red: TrackmateXML, tm_green: TrackmateXML, metric: pd.DataFrame):
        super().__init__(tm_red, tm_green)
        self.metric_df = metric.sort_values("metric").reset_index(drop=True)

    @cache
    def calculate_metric(self, green_track_id, red_track_id):
        return self.metric_df.query("green_track == @green_track_id and red_track == @red_track_id").metric.item()


def calculate_positional_derivative(df):
    """Calculate the positional derivative for the track d(xy)/dt and represent in polar coordinates"""
    df = df.sort_values("frame")
    result_df = pd.DataFrame()
    result_df["d_frame"] = df.frame.diff()
    result_df["d_x"] = df.POSITION_X.diff()
    result_df["d_y"] = df.POSITION_Y.diff()
    # result_df.dropna()
    result_df["magnitude"] = (result_df.d_x**2 + result_df.d_y**2) ** 0.5
    result_df["angle"] = np.arctan2(
        (result_df.d_y / result_df.magnitude).values.astype(np.float32),
        (result_df.d_x / result_df.magnitude).values.astype(np.float32),
    )

    return result_df
