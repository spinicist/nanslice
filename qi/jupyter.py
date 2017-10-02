#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from .image import colorize
from .box import Box
from .slice import Slice

def static(img, cmap='gray', bbox=None, point=None):
    if not bbox:
        bbox = Box(img, mask=True)
    if not point:
        point = bbox.center

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        sl = Slice(bbox, point, i, 256, orient='clin')
        sl_img = sl.sample(img, order=0)
        im = axes[i].imshow(sl_img, origin='lower', extent=sl.extent, cmap=cmap, vmin = 0.1)
        axes[i].axis('off')
        if i == 2:
            fig.colorbar(im)
    return (fig, axes)

def interactive(img, cmap='gray', window=(2, 98)):
    import ipywidgets as ipy

    # Get some information about the image
    bbox = Box(img, mask=True)
    window_vals = np.nanpercentile(img.get_data(), window)
    print(window_vals, np.max(img.get_data()), np.min(img.get_data()))
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    implots = [None, None, None]
    for i in range(3):
        sl = Slice(bbox, bbox.center, i, 256, orient='clin')
        sl_img = sl.sample(img, order=0)
        sl_color = colorize(sl_img, cmap, window_vals)
        implots[i] = axes[i].imshow(sl_color, origin='lower', extent=sl.extent, cmap=cmap, vmin = 0.1)
        axes[i].axis('off')

    def wrap_sections(x, y, z):
        for i in range(3):
            sl = Slice(bbox, np.array((x, y, z)), i, 256, orient='clin')
            sl_img = sl.sample(img, order=0)
            sl_color = colorize(sl_img, cmap, window_vals)
            implots[i].set_data(sl_color)
            plt.show()

    # Setup widgets
    slider_x = ipy.FloatSlider(min=bbox.start[0], max=bbox.end[0], value=bbox.center[0])
    slider_y = ipy.FloatSlider(min=bbox.start[1], max=bbox.end[1], value=bbox.center[1])
    slider_z = ipy.FloatSlider(min=bbox.start[2], max=bbox.end[2], value=bbox.center[2])
    widgets = ipy.interactive(wrap_sections, x=slider_x, y=slider_y, z=slider_z)

    # Now do some manual layout
    hbox = ipy.HBox(widgets.children[0:3]) # Set the sliders to horizontal layout
    vbox = ipy.VBox((hbox, widgets.children[3]))
    # iplot.widget.children[-1].layout.height = '350px'
    display(vbox)