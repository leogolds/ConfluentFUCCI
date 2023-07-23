"""
Load sample data.
"""
import importlib.metadata
from importlib import metadata
from pathlib import Path

import pandas
import pooch

try:
    ver = 'v' + metadata.version("confluentfucci")
except importlib.metadata.PackageNotFoundError:
    ver = 'v1.0.0+12.do9iwd'

file_registry = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("confluentfucci"),
    # The remote data is on Github
    base_url="https://github.com/leogolds/ConfluentFUCCI/raw/{version}",
    # base_url="https://github.com/rick/plumbus/raw/{version}/data/",
    version=ver,
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry={
        "models/cellpose/nuclei_red_v2": "1767fd32b838ae1ee46759afd85a4bcb433ef0ee3ad35e2cefa81edf02a889bb",
        "models/cellpose/nuclei_green_v2": "64d61f45be0e3802c868e89f4f709c60684c99e582e6f442c39704f20721c02a",
        "models/trackmate/basic_settings.xml": "c7a6ac14a92a4e4de671bcd8ca36fc8f08c95e611d8783bd475a5d8ebcbd2d6e",
        "data/3_frames/red.tif": "d654630f91400b11bc0b2eec4620b54ba8e9b2a02285706614158b2cec75821e",
        "data/3_frames/green.tif": "8ceba2f6d54c4b9fef2ca2ffdec07fcbec4d98ca8c8b395c3cff56945f205f5a",
        "data/3_frames/phase.tif": "8d8043f6f8f1403bd93755365db01d11fad471e0bbb5b1528a923755bdf5d6e9",
        "data/60_frames/red.tif": "24eba5b3e2b8376e6bfdce1206ee1c56bd444127872c0ff7af6015d903de1553",
        "data/60_frames/green.tif": "102cf8edd9bb98a860e6cc6ddccc312508d3892be884c6749d34afdc0a75e535",
        "data/60_frames/phase.tif": "6dc1d8ff4b8282a495680ab0a7d093bf7528845daf6f49301404ef9045b8d562",
    },
)


def fetch_short_example_data():
    short_data = [path for path in file_registry.registry_files if 'data/3_frames' in path]
    short_data_paths = [Path(file_registry.fetch(path, progressbar=True)) for path in short_data]

    return short_data_paths


def fetch_long_example_data():
    short_data = [path for path in file_registry.registry_files if 'data/60_frames' in path]
    short_data_paths = [Path(file_registry.fetch(path, progressbar=True)) for path in short_data]

    return short_data_paths


def fetch_red_model():
    """
    Fetch the red channel CellPose model
    """
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = file_registry.fetch("models/cellpose/nuclei_red_v2", progressbar=True)
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    return Path(fname)


def fetch_green_model():
    """
    Fetch the red channel CellPose model
    """
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = file_registry.fetch("models/cellpose/nuclei_green_v2", progressbar=True)
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    return Path(fname)


def fetch_trackmate_settings():
    """
    Fetch the red channel CellPose model
    """
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = file_registry.fetch("models/trackmate/basic_settings.xml", progressbar=True)
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    return Path(fname)
