from pathlib import Path
import holoviews as hv
import pandas as pd
import numpy as np

from confluentfucci import utils

import napari
from PIL import Image
import numpy as np


img = Image.open(r'C:\Users\leo\PycharmProjects\MicroscopyPipeline\experiments\exp_01/img_50_rgb.png')
img = np.array(img)
# add the image
microns_per_pixel = [0.67, 0.67]
base_data_path = Path(r"D:\Data\full_pipeline_tests\left_60_frames")

phase = utils.read_stack(base_data_path / "phase.tif")[50,...]

viewer = napari.view_image(np.flipud(phase), scale=microns_per_pixel)
viewer.add_image(img, scale=microns_per_pixel, opacity=0.2)
viewer.scale_bar.visible = False
viewer.scale_bar.unit = "um"
viewer.camera.zoom = 2


# def reverse_vertices():
#     all_verts = []
#     for verts in filtered_vor_stats_df.query("area < area.quantile(0.99)").query("frame == @frame").vertices.values:
#         all_verts.append([a[::-1] for a in verts])
#
#     return all_verts
#
#
# rev_verts = reverse_vertices()

# add the polygons
# shapes_layer = viewer.add_shapes(polygons, shape_type='polygon', edge_width=5,
#                           edge_color='coral', face_color='royalblue')
#viewer.add_shapes( rev_verts, shape_type='polygon', edge_width=2, opacity=0.4,
#                          edge_color='coral', face_color='royalblue', scale=microns_per_pixel)
napari.run()


# red_segmented_stack = utils.read_stack(base_data_path / "red_segmented.tiff")
# green_segmented_stack = utils.read_stack(base_data_path / "green_segmented.tiff")
# phase = utils.read_stack(base_data_path / "green_segmented.tiff")

# from PIL import Image
# import numpy as np
# from skimage.exposure import equalize_adapthist


# def save_rgb_on_phase_overlay(frame=0, name='img'):
#     rgb = np.dstack([red_segmented_stack[frame] != 0,
#                      green_segmented_stack[frame] != 0,
#                      np.zeros(shape=green_segmented_stack.shape)[frame]])
#     print(f'{rgb.shape=}')
#
#     rgbArray = np.zeros(rgb.shape, 'uint8')
#     rgbArray[..., 0] = (red_segmented_stack[frame] != 0).astype(np.uint8) * 255
#     rgbArray[..., 1] = (green_segmented_stack[frame] != 0).astype(np.uint8) * 255
#     rgbArray[..., 2] = 0 * 256
#     img = Image.fromarray(np.flipud(rgbArray))
#     img.save(f'{name}_{frame}_rgb.png')
#
#     opts = {
#         # "aspect": rgb.shape[1]/rgb.shape[0],
#         # "aspect": 988/404,
#         "invert_yaxis": False,
#         "responsive": False,
#     }
#     bounds = (0, 0, rgb.shape[1], rgb.shape[0])
#
#     # img = equalize_adapthist(phase_path[frame,...])
#     # img = np.expand_dims(img, axis=0)
#     # img.shape
#     # base_layout = utils.view_stacks([img], frame=0)
#     vor = draw_voronoi_tiling(
#         filtered_vor_stats_df.query("area < area.quantile(0.99)")
#         , frame=frame).opts(**opts2)
#     hv.save(vor, 'voronoi.png')
#     my_figure = hv.RGB.load_image(f'voronoi.png') * hv.RGB.load_image(f'{name}_{frame}_rgb.png', bounds=bounds).opts(
#         alpha=0.4, **opts)
#
#     hv.save(my_figure, f'{name}_{frame}_voronoi_overlay.png')
#
#     returnve_rgb_on_phase_overlay(frame=0, name='img'):
#     rgb = np.dstack([red_segmented_stack[frame] != 0,
#                      green_segmented_stack[frame] != 0,
#                      np.zeros(shape=green_segmented_stack.shape)[frame]])
#     print(f'{rgb.shape=}')
#
#     rgbArray = np.zeros(rgb.shape, 'uint8')
#     rgbArray[..., 0] = (red_segmented_stack[frame] != 0).astype(np.uint8) * 255
#     rgbArray[..., 1] = (green_segmented_stack[frame] != 0).astype(np.uint8) * 255
#     rgbArray[..., 2] = 0 * 256
#     img = Image.fromarray(np.flipud(rgbArray))
#     img.save(f'{name}_{frame}_rgb.png')
#
#     opts = {
#         # "aspect": rgb.shape[1]/rgb.shape[0],
#         # "aspect": 988/404,
#         "invert_yaxis": False,
#         "responsive": False,
#     }
#     bounds = (0, 0, rgb.shape[1], rgb.shape[0])
#
#     # img = equalize_adapthist(phase_path[frame,...])
#     # img = np.expand_dims(img, axis=0)
#     # img.shape
#     # base_layout = utils.view_stacks([img], frame=0)
#     vor = draw_voronoi_tiling(
#         filtered_vor_stats_df.query("area < area.quantile(0.99)")
#         , frame=frame).opts(**opts2)
#     hv.save(vor, 'voronoi.png')
#     my_figure = hv.RGB.load_image(f'voronoi.png') * hv.RGB.load_image(f'{name}_{frame}_rgb.png', bounds=bounds).opts(
#         alpha=0.4, **opts)
#
#     hv.save(my_figure, f'{name}_{frame}_voronoi_overlay.png')
#
#     return my_figure