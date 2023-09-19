# from __future__ import annotations

import os
import sys

from fiji.plugin.trackmate import (
    Logger,
    TrackMate,
)
from fiji.plugin.trackmate.io import (
    TmXmlReader,
    TmXmlWriter,
)
from ij import IJ
from java.io import File
from java.lang import System

# We have to do the following to avoid errors with UTF8 chars generated in
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding("utf-8")

# -----------------
# Read data stack
# -----------------

# Get currently selected image
if os.environ.get("DOCKER"):
    data_path = os.environ.get("DOCKER_TIFF_STACK")
else:
    data_path = "/data/" + os.environ.get("TIFF_STACK")

print("reading data from: " + data_path)
imp = IJ.openImage(data_path)
print("data read successfully")

# -----------------
# Read settings from XML
# -----------------

if os.environ.get("DOCKER"):
    settings_path = os.environ.get("DOCKER_SETTINGS_XML")
else:
    settings_path = "/settings/" + os.environ.get("SETTINGS_XML")
print("Reading settings from: " + settings_path)
file = File(settings_path)
reader = TmXmlReader(file)
if not reader.isReadingOk():
    sys.exit(reader.getErrorMessage())
print("Settings read successfully")
# -----------------
# Get a full model
# -----------------

model = reader.getModel()
model.setLogger(Logger.IJ_LOGGER)


settings = reader.readSettings(imp)

# -------------------
# Instantiate plugin
# -------------------

trackmate = TrackMate(model, settings)

# --------
# Process
# --------
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

# Echo results with the logger we set at start:
model.getLogger().log(str(model))

# --------
# Write Results
# --------
print("writing results.xml")

if os.environ.get("DOCKER"):
    f = File(os.environ.get("DOCKER_TIFF_STACK") + ".xml")
else:
    f = File("/data/" + os.environ.get("TIFF_STACK") + ".xml")
xml_writer = TmXmlWriter(f)

xml_writer.appendModel(model)
xml_writer.appendSettings(settings)
xml_writer.writeToFile()
print("finished writing results.xml")


print("Analysis complete. Exiting...")
System.exit(0)
