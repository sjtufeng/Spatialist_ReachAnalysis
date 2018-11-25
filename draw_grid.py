"""Draw a street grid based on the imported CSV file.

Author: Chen Feng
Last updated on Nov. 21, 2018
"""

from __future__ import division
import numpy as np
from pyx import *
import os

os.chdir(r'C:\_SoftwareDevelopment\Grasshopper\GhPython_PatternGeneration\data_RealExamples\Apt')
csv_file = "Apt_SGA_6sep2018.csv"

na = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

# Set canvas for drawing
c = canvas.canvas()
stroke_width = 0.02
scale_factor = 1 / 50
for i in range(len(na)):
    line = path.line(na[i][0] * scale_factor, na[i][1] * scale_factor, na[i][2] * scale_factor, na[i][3] * scale_factor)
    c.stroke(line, [style.linewidth(stroke_width), color.rgb.black])

    # Export the drawing to a PDF file
    os.chdir("C:/Users/cfeng/Desktop/Outbox")
    c.writePDFfile('segment_map_snapshot')
