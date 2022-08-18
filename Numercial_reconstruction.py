import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

import tensorflow as tf
from tensorflow.keras import layers as ls, activations as acts
import tensorflow_addons as tfa

from skimage.restoration import unwrap_phase
from PIL import Image

from fringe.utils.io import import_image, export_image
from fringe.utils.modifiers import ImageToArray, ConvertToTensor
from fringe.process.gpu import AngularSpectrumSolver as AsSolver

folder_path = '/Users/chuckles/Desktop/Holographic Reconstruction/test image/'
file_name = 'target512_2'
hologram_path = folder_path+file_name+'.jpg'

p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')

hologram = import_image(hologram_path, modifiers=[p1])
hologram_amp = tf.math.abs(hologram)

def Scale(img, perc, max_val):
	img *= perc
	img += 1 - perc
	img /= max_val
	return img
  
# Adjusting contrast
hologram_amp = Scale(hologram_amp, perc=1, max_val=np.max(hologram_amp))

solver = AsSolver(shape=hologram_amp.shape, dx=1.12, dy=1.12, wavelength=650e-3)

def reconstruct(z):
    rec = solver.solve(hologram, z)
    amp = np.abs(rec)
    phase = unwrap_phase(np.angle(rec))
    return amp

init_z = 195
amp = hologram_amp

fig, ax = plt.subplots()
plt.imshow(reconstruct(init_z), cmap='gray')
ax.set_xlabel('z height = ' + str(init_z))

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a vertically oriented slider to control the amplitude
axz = plt.axes([0.1, 0.25, 0.0225, 0.63])
z_slider = Slider(
    ax=axz,
    label="Z-height",
    valmin=8400, # for thin layer 100 to 400
    valmax=8600, # 
    valstep= 10,
    valinit=init_z,
    orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    z = z_slider.val
    ax.set_xlabel('z height = ' + str(z))
    global amp
    amp = reconstruct(z)  
    ax.imshow(amp, cmap='gray')
    fig.canvas.draw_idle()

ax_save = plt.axes([0.8, 0.025, 0.1, 0.04])
save_bnt = Button(ax=ax_save, label='Save')

def save(event):
    i = Image.fromarray(amp.numpy(), "RGB")
    export_image(i, folder_path+file_name+'recon'+'.jpg')

    print('Saved image')

save_bnt.on_clicked(save)

# register the update function with each slider
z_slider.on_changed(update)

plt.show()