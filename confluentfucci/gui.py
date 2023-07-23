import sys
import traceback
from pathlib import Path
from time import sleep
from tkinter import (
    Tk,
    filedialog,
)

import cellpose.core
import holoviews as hv
import hvplot.pandas
import napari
import numpy as np
import pandas as pd
import panel as pn
import param
from dask_image.imread import imread
from shapely import Polygon
from skimage.exposure import equalize_adapthist

from confluentfucci import data
from confluentfucci.math import (
    CartesianSimilarity,
    CartesianSimilarityFromFile,
    TrackmateXML,
    compute_voronoi,
    compute_voronoi_stats,
    filter_voronoi_tiling,
)
from confluentfucci.utils import (
    get_docker_client,
    read_stack,
    run_trackmate,
    segment_stack,
)

pn.extension("terminal")


class CollectiveStats:
    def __init__(self, metric: CartesianSimilarity, phase_stack_path: Path):
        self.stack = read_stack(phase_stack_path)
        _, y, x = self.stack.shape
        self.image_rect = Polygon([(0, 0), (0, y), (x, y), (x, 0)])

        self.metric = metric
        self.instance = None

        self.flow_frame = pn.widgets.IntSlider(name="Frame", value=1, start=1, end=self.stack.shape[0] - 1)
        self.flow_min_magnitude = pn.widgets.IntSlider(name="Min Magnitude", value=1, start=0, end=25)

        flow_field_df = self.metric.calculate_flow_field(shape=self.stack.shape[1:])
        flow_field_df["magnitude_um"] = flow_field_df.magnitude * 0.67

        self.flow = pn.bind(
            visualize_flow_field,
            flow_field_df=flow_field_df,
            # flow_field_df=t,
            frame=self.flow_frame.param.value_throttled,
            min_magnitude=self.flow_min_magnitude.param.value_throttled,
            red_stack=self.stack,
        )

        self.count_color_select = pn.widgets.CheckButtonGroup(
            value=["red", "yellow", "green"],
        )

        self.sidebar = self.get_sidebar()
        self.main = pn.GridSpec(mode="override", sizing_mode="stretch_width", height=600)

        self.template = self.get_template()

        self.sidebar.param.watch(self.update_main, ["active"], onlychanged=True)

        self.sidebar.active = [0]

    def get_template(self):
        template = pn.template.FastListTemplate(
            title="PyFucciTrack",
            theme_toggle=False,
            sidebar=[self.sidebar],
            main=self.main,
        )

        return template

    def get_sidebar(self):
        sidebar = pn.Accordion(toggle=True)

        sidebar.append(("Counts", pn.Column("Explanation", self.count_color_select)))
        sidebar.append(("Area Estimate", "Explanation"))
        sidebar.append(
            (
                "Flow Field",
                pn.Column("Explanation", self.flow_frame, self.flow_min_magnitude),
            )
        )

        return sidebar

    def update_main(self, event):
        # print(event)
        if event.new[0] == 0:
            self.main[:, :3], self.main[:, 3:6] = self.get_count()
        elif event.new[0] == 1:
            self.main[:, :] = self.get_area_estimate()
        elif event.new[0] == 2:
            self.main[:, :] = self.flow
        else:
            self.main[:, :] = "other"

    def show(self):
        self.template.show()

    def get_count(self):
        df = self.metric.count_cells_in_bins()

        df = df.groupby("frame")[["red", "yellow", "green"]].sum()
        df["red_norm"] = df.red / df[["green", "red", "yellow"]].sum(axis="columns")
        df["green_norm"] = df.green / df[["green", "red", "yellow"]].sum(axis="columns")
        df["yellow_norm"] = df.yellow / df[["green", "red", "yellow"]].sum(axis="columns")

        df["time_hours"] = df.index / 6

        a = df.hvplot.line(
            ylabel="Count (abs.)",
            x="time_hours",
            xlabel="Time (hours)",
            y=["red", "yellow", "green"],
            title="absolute",
            line_color=["red", "yellow", "green"],
        ).opts(show_legend=False, shared_axes=False)

        b = df.hvplot.line(
            ylabel="Count (norm.)",
            x="time_hours",
            xlabel="Time (hours)",
            y=["red_norm", "yellow_norm", "green_norm"],
            title="proportional",
            line_color=["red", "yellow", "green"],
        ).opts(show_legend=False, shared_axes=False)

        return a, b

    def get_area_estimate(self):
        df = self.metric.partition_cells_into_bins()
        valid_regions_df = (
            # df.groupby("timestep").progress_apply(compute_voronoi).query("valid_region")
            df.groupby("frame")
            .apply(compute_voronoi)
            .query("valid_region")
        )

        # vor_stats_df = valid_regions_df.groupby(["frame", "bin"]).apply(compute_voronoi_stats)
        vor_stats_df = valid_regions_df.groupby(["frame"]).apply(compute_voronoi_stats)
        filtered_vor_stats_df = filter_voronoi_tiling(vor_stats_df, self.image_rect)
        df = (
            filtered_vor_stats_df.groupby(["frame", "color"])["area"]
            .agg(["mean", "sem", "count"])
            .unstack()
            .fillna(0)
            .stack()
            .reset_index()
        )
        df["ci_min"] = df["mean"] - 1.96 * df["sem"]
        df["ci_max"] = df["mean"] + 1.96 * df["sem"]

        df["time_hours"] = df.frame / 6

        fig = (
            df.hvplot.line(
                ylabel="area (um^2)",
                x="time_hours",
                y="mean",
                xlabel="Time (hours)",
                # title="Cellular area (95% CI)",
                by="color",
                line_color=[
                    "green",
                    "red",
                    "yellow",
                ],
            )
            * df.query('color == "red"')
            .hvplot.area(
                x="time_hours",
                y="ci_min",
                y2="ci_max",
                alpha=0.3,
            )
            .opts(color="red")
            * df.query('color == "yellow"')
            .hvplot.area(
                x="time_hours",
                y="ci_min",
                y2="ci_max",
                alpha=0.3,
            )
            .opts(color="yellow")
            * df.query('color == "green"')
            .hvplot.area(
                x="time_hours",
                y="ci_min",
                y2="ci_max",
                alpha=0.3,
            )
            .opts(color="green")
        )
        fig.opts(show_legend=False)

        return fig


def select_files_model():
    red_model, green_model = data.fetch_red_model(), data.fetch_green_model()
    root = Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)
    files = filedialog.askopenfilename(initialdir=red_model.parent)
    print(files)
    return Path(files)


def view_segmented(data_dir_path):
    base_data_path = Path(data_dir_path)
    viewer = napari.Viewer(title="PyFucciTrack Viewer")

    red = imread(base_data_path / "red.tif")
    green = imread(base_data_path / "green.tif")

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

    red_seg = (imread(base_data_path / "red_segmented.tiff") > 0).compute()
    green_seg = (imread(base_data_path / "green_segmented.tiff") > 0).compute()

    viewer.add_labels(red_seg, blending="additive", color={1: "red"})
    viewer.add_labels(green_seg, blending="additive", color={1: "green"})

    viewer.layers[:] = viewer.layers[::-1]

    napari.run(force=True, gui_exceptions=True)


def trackmate_available():
    try:
        get_docker_client().images.pull("leogold/trackmate:v1")
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False


def check_cellpose_gpu():
    return cellpose.core.use_gpu()


def check_docker():
    try:
        get_docker_client().ping()
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False


class AppUI(param.Parameterized):
    data_dir_path = param.Path()
    red_path = param.Path()
    green_path = param.Path()
    phase_path = param.Path()
    red_model_path = param.Path()
    green_model_path = param.Path()
    analysis_available = param.Boolean(default=False)

    docker_check = param.Boolean(default=False)
    gpu_check = param.Boolean(default=False)
    trackmate_check = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        self.validate_btn = pn.widgets.Button(name="Validate Install", button_type="danger")
        self.validate_btn.on_click(self.validate_install)
        pn.state.onload(self.validate_install)

        self.select_data_path_btn = pn.widgets.Button(name="Select Data Path", button_type="primary")
        self.select_data_path_btn.on_click(self.select_data_folder)
        self.select_red_cellpose_model_btn = pn.widgets.Button(name="Select Red CellPose Model", button_type="primary")
        self.select_red_cellpose_model_btn.on_click(self.select_red_model)
        self.select_green_cellpose_model_btn = pn.widgets.Button(
            name="Select Green CellPose Model", button_type="primary"
        )
        self.select_green_cellpose_model_btn.on_click(self.select_green_model)

        self.segment_one_btn = pn.widgets.Button(name="Segment one frame", button_type="primary")
        self.segment_one_btn.on_click(self.segment_one)
        self.segment_all_btn = pn.widgets.Button(name="Segment stack", button_type="primary")
        self.segment_all_btn.on_click(self.segment_stack)
        self.segmentation_progress = pn.widgets.Tqdm(width=300)

        self.run_tracking_btn = pn.widgets.Button(name="Track", button_type="primary")
        self.run_tracking_btn.on_click(self.track)
        self.tracking_terminal = pn.widgets.Terminal()

        self.run_analysis_btn = pn.widgets.Button(name="Analyze", button_type="primary")
        self.run_analysis_btn.on_click(self.run_analysis)
        self.analysis_progress = pn.widgets.Tqdm(width=300)
        self.analysis_progress.pandas(desc="anslysis progress")
        self.analysis_ui = None

        self.anslysis_tabs = pn.Tabs(
            ("Counts", self.get_counts),
            ("Area", self.get_area),
            ("Flow", self.get_flow),
        )

        self.sidebar = self.get_sidebar()
        self.main = pn.GridSpec(mode="override", sizing_mode="stretch_width", height=600)

        self.template = self.get_template()

        self.sidebar.param.watch(self.update_main, ["active"], onlychanged=True)

        self.sidebar.active = [0]

    def validate_install(self, event=None):
        self.validate_btn.disabled = True

        self.docker_check, self.trackmate_check, self.gpu_check = False, False, False

        self.docker_check = True if check_docker() else False
        self.trackmate_check = True if trackmate_available() else False
        self.gpu_check = True if check_cellpose_gpu() else False

        sleep(2)
        if all([self.docker_check, self.trackmate_check, self.gpu_check]):
            self.validate_btn.button_type = "success"
        else:
            self.validate_btn.button_type = "danger"
        self.validate_btn.disabled = False

    def select_data_folder(self, *b):
        root = Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        short_data, long_data = data.fetch_short_example_data(), data.fetch_long_example_data()
        files = filedialog.askdirectory(initialdir=short_data[0].parent.parent)
        # files = filedialog.askdirectory(
        #     initialdir=r"D:\Data\full_pipeline_tests\left_60_frames"
        # )

        # print(files)
        self.data_dir_path = Path(files)

        red_path = Path(self.data_dir_path) / "red.tif"
        green_path = Path(self.data_dir_path) / "green.tif"
        phase_path = Path(self.data_dir_path) / "phase.tif"

        self.red_path = red_path if red_path.exists() else None
        self.green_path = green_path if green_path.exists() else None
        self.phase_path = phase_path if phase_path.exists() else None

        if all([red_path.exists(), green_path.exists(), phase_path.exists()]):
            self.select_data_path_btn.button_type = "success"
        else:
            self.select_data_path_btn.button_type = "danger"

        return files

    def select_red_model(self, _):
        self.red_model_path = select_files_model()
        self.select_red_cellpose_model_btn.button_type = "success"

    def select_green_model(self, _):
        self.green_model_path = select_files_model()
        self.select_green_cellpose_model_btn.button_type = "success"

    def track(self, event):
        self.run_tracking_btn.disabled = True
        sys.stdout = self.tracking_terminal

        (Path(self.data_dir_path) / "metric.h5").unlink(missing_ok=True)
        run_trackmate(
            data.fetch_trackmate_settings(),
            Path(self.data_dir_path) / "red_segmented.tiff",
        )
        run_trackmate(
            data.fetch_trackmate_settings(),
            Path(self.data_dir_path) / "green_segmented.tiff",
        )

        sys.stdout = sys.__stdout__
        self.run_tracking_btn.disabled = False

    def segment_one(self):
        pass

    def segment_stack(self, event):
        segment_stack(
            path=Path(self.red_path),
            model=Path(self.red_model_path),
            panel_tqdm_instance=self.segmentation_progress,
        )
        segment_stack(
            path=Path(self.green_path),
            model=Path(self.green_model_path),
            panel_tqdm_instance=self.segmentation_progress,
        )

        self.main[:, :] = "# Segmentation\nPlease close viewer to continue"

        view_segmented(self.data_dir_path)

    @param.depends("analysis_available")
    def get_counts(self):
        if self.analysis_available:
            gspec = pn.GridSpec(sizing_mode="stretch_width", height=600)
            gspec[:, :3], gspec[:, 3:6] = self.analysis_ui.get_count() if self.analysis_ui else (None, None)
            return gspec
        else:
            return None

    @param.depends("analysis_available")
    def get_area(self):
        return pn.bind(self.analysis_ui.get_area_estimate) if self.analysis_available else None

    @param.depends("analysis_available")
    def get_flow(self):
        return self.analysis_ui.flow if self.analysis_available else None

    def run_analysis(self, event):
        self.run_analysis_btn.disabled = True
        self.analysis_available = False
        tm_red = TrackmateXML(Path(self.data_dir_path) / "red_segmented.tiff.xml")
        tm_green = TrackmateXML(Path(self.data_dir_path) / "green_segmented.tiff.xml")

        metric_path = Path(self.data_dir_path) / "metric.h5"
        if metric_path.exists():
            metric_df = pd.read_hdf(metric_path, key="metric")
            self.metric = CartesianSimilarityFromFile(tm_red, tm_green, metric_df)
        else:
            self.metric = CartesianSimilarity(tm_red, tm_green)
            metric_df = self.metric.calculate_metric_for_all_tracks()
            metric_df.to_hdf(metric_path, key="metric")

        self.analysis_ui = CollectiveStats(self.metric, Path(self.data_dir_path) / "phase.tif")

        self.analysis_available = True
        self.run_analysis_btn.disabled = False

    def get_sidebar(self):
        sidebar = pn.Accordion(toggle=True, sizing_mode="stretch_width")

        sidebar.append(
            (
                "Welcome",
                pn.Column(self.validate_btn),
            )
        )
        sidebar.append(
            (
                "Input",
                pn.Column(
                    self.select_data_path_btn,
                    self.select_red_cellpose_model_btn,
                    self.select_green_cellpose_model_btn,
                ),
            )
        )
        sidebar.append(
            (
                "Segmentation",
                pn.Column(
                    self.segment_one_btn,
                    self.segment_all_btn,
                    self.segmentation_progress,
                ),
            )
        )
        sidebar.append(
            (
                "Tracking",
                pn.Column(
                    self.run_tracking_btn,
                ),
            )
        )
        sidebar.append(
            (
                "Analysis",
                pn.Column(
                    self.run_analysis_btn,
                    self.analysis_progress,
                ),
            )
        )

        return sidebar

    def get_template(self):
        template = pn.template.BootstrapTemplate(
            title="ConfluentFUCCI",
            theme_toggle=False,
            sidebar=[self.sidebar],
            main=self.main,
        )

        return template

    @param.depends("docker_check", "trackmate_check", "gpu_check")
    def get_welcome_message(self):
        # Docker: {🕑 if self.docker_check is None or (✅ if self.docker_check else ❌)}

        text = pn.pane.Markdown(
            f"""# Welcome
See you soon
                
## Status check 
GPU: {'✅' if self.gpu_check else '❌'} {'See [link](example.com) for advice' if not self.gpu_check else ''}

Docker: {'✅' if self.docker_check else '❌'} {'See [link](example.com) for advice' if not self.docker_check else ''}

TrackMate: {'✅' if self.trackmate_check else '❌'} {'See [link](example.com) for advice' if not self.trackmate_check else ''}
""",
            dedent=True,
        )
        return text

    def update_main(self, event):
        # print(event)
        if event.new[0] == 0:
            self.main[:, :] = self.get_welcome_message
        elif event.new[0] == 1:
            self.main[:, :] = pn.Column(
                f"# input",
                "## Data",
                # pn.Param(self.param.data_dir_path),
                pn.Param(self.param.red_path),
                pn.Param(self.param.green_path),
                pn.Param(self.param.phase_path),
                "## CellPose Models",
                pn.Param(self.param.red_model_path),
                pn.Param(self.param.green_model_path),
            )
        elif event.new[0] == 2:
            self.main[:, :] = "# Segmentation"
        elif event.new[0] == 3:
            self.main[:, :] = pn.Column("# Tracking", self.tracking_terminal)
        elif event.new[0] == 4:
            self.main[:, :] = pn.Column("# Analysis", self.anslysis_tabs)
        else:
            self.main[:, :] = "other"

    def show(self):
        self.template.show()


def visualize_flow_field(flow_field_df, red_stack, frame=30, min_magnitude=0):
    aspect = red_stack.shape[1] / red_stack.shape[2]
    figure_width = 300

    frame_df = flow_field_df.dropna().query("frame == @frame")

    df = frame_df.query("magnitude > @min_magnitude")

    v_line = hv.VLine(
        x=min_magnitude,
    ).opts(color="red")
    histograms = (
        (
            frame_df.magnitude.hvplot.hist(title="Velocity", xlabel="velocity (um/frame)", xlim=(0, 20)) * v_line
            + df.angle.hvplot.hist(title="Direction", xlabel="angle (rad)", xlim=(-np.pi, np.pi))
        )
        .cols(1)
        .opts(shared_axes=False)
    )

    a = hv.Image(
        equalize_adapthist(np.flipud(red_stack[frame, ...])),
        bounds=(0, 0, red_stack.shape[2], red_stack.shape[1]),
    ).opts(
        frame_width=red_stack.shape[2],
        frame_height=red_stack.shape[1],
        cmap="gray",
        xticks=0,
        yticks=0,
    ) * df.hvplot.vectorfield(
        x="x_bin",
        y="y_bin",
        mag="magnitude_um",
        angle="angle"
        # x="x_bin", y="y_bin", mag="magnitude", angle="angle"
    ).opts(
        # color="magnitude",
        color="magnitude_um",
        xlim=(0, red_stack.shape[2]),
        ylim=(0, red_stack.shape[1]),
        frame_width=red_stack.shape[2],
        frame_height=red_stack.shape[1],
        colorbar=True,
        cmap="bgy",
    ).redim.range(
        magnitude_um=(0, 20)
    )

    return pn.Row(
        a.opts(frame_width=figure_width, frame_height=int(figure_width * aspect)),
        histograms,
        height=int(figure_width * aspect),
    )


# magnification_towards_camera = 1
# # pixel_size_in_microns = 0.345 * magnification_towards_camera
# pixel_size_in_microns = 0.67 * magnification_towards_camera
# calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2

# AppUI().template.servable()
# AppUI().template.show()

# pn.Row(
#     pn.widgets.Button(
#         icon="alert-triangle-filled", button_type="warning", name="WARNING"
#     ),
#     pn.widgets.Button(icon="bug", button_type="danger", name="Error"),
# ).servable()

if __name__ == "__main__":
    AppUI().get_template().show()
    # magnification_towards_camera = 1
    # # pixel_size_in_microns = 0.345 * magnification_towards_camera
    # pixel_size_in_microns = 0.67 * magnification_towards_camera
    # calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2
    #
    # # base_data_path = Path("data/fucci_60_frames")
    # base_data_path = Path(r"D:\Data\full_pipeline_tests\left_60_frames")
    # red_stack = base_data_path / "red.tif"
    # green_stack = base_data_path / "green.tif"
    # phase_stack = base_data_path / "phase.tif"
    #
    # base_model_path = Path(r"D:\Data\full_pipeline_tests\fuccitrack_data\models")
    # red_model = base_model_path / "fuccitrack_data_red"
    # green_model = base_model_path / "fuccitrack_data_green"
    #
    # # segment_stack(red_stack, red_model)
    # # segment_stack(green_stack, green_model)
    #
    # settings_xml = Path(r"models/trackmate/basic_settings.xml")
    # red_data_stack = base_data_path / "red_segmented.tiff"
    # green_data_stack = base_data_path / "green_segmented.tiff"
    #
    # # run_trackmate(settings_xml, red_data_stack)
    # # run_trackmate(settings_xml, green_data_stack)
    #
    # tm_red = trackmate_utils.TrackmateXML(base_data_path / "red_segmented.tiff.xml")
    # tm_green = trackmate_utils.TrackmateXML(base_data_path / "green_segmented.tiff.xml")
    #
    # metric_df = pd.read_hdf(base_data_path / "metric.h5", key="metric")
    # metric = trackmate_utils.CartesianSimilarityFromFile(tm_red, tm_green, metric_df)
    #
    # CollectiveStats(metric, phase_stack).show()
