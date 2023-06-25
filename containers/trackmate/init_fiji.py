from __future__ import annotations

import imagej
import imagej.doctor

imagej.doctor.debug_to_stderr()
ij = imagej.init('/opt/fiji/Fiji.app')
print(f'ImageJ version: {ij.getVersion()}')
