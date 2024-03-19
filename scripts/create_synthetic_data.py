# Creat synthetic data for testing pixel shift
import numpy as np
import pyidi

def global_rotation(x0, y0, frequency, amplitude, t):
    x = np.cos(2 * np.pi * frequency * t) * amplitude + x0
    y = np.sin(2 * np.pi * frequency * t) * amplitude + y0
    return x, y

image_width     = 1024,
image_height    = 512
x0              = image_width/2
y0              = image_height/2
amplitude       = min(image_width, image_height)/4
fps             = 40000.0
total_time      = 0.25
number_of_rotations = 1
frequency       = number_of_rotations/total_time
total_frames    = int(total_time * fps)

mraw = np.zeros((total_frames, image_height, image_width), dtype=np.memmap)