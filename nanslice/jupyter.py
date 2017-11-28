#!/usr/bin/env python
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from . import util
from .box import Box
from .slice import Slice

Options = namedtuple("Options",
                     "interp_order color_map color_lims color_scale color_mask_thresh alpha_lims")
def static(img, cmap='gray', bbox=None, point=None):
    if not bbox:
        bbox = Box.fromMask(img)
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

def interactive(img, img_cmap='gray', img_window=(2, 98), mask=None,
                color_img=None, color_cmap='viridis', color_window=None, color_thresh=None,
                alpha_img=None, alpha_window=None,
                orient='clin', samples=128):
    import ipywidgets as ipy

    # Get some information about the image
    if mask:
        bbox = Box.fromMask(mask)
    else:
        bbox = Box.fromImage(img)
    window_vals = np.nanpercentile(img.get_data(), img_window)
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor='r')
    implots = [None, None, None]
    init = False

    if color_img and color_window is None:
        color_window = np.nanpercentile(color_img.get_data(), (2, 98))
    if alpha_img and alpha_window is None:
        alpha_window = np.nanpercentile(alpha_img.get_data(), (2, 98))
    
    options = Options(interp_order=0, color_map=color_cmap, color_lims=color_window, color_scale=1,
                      color_mask_thresh=color_thresh,
                      alpha_lims=alpha_window)

    def wrap_sections(X, Y, Z):
        for i in range(3):
            sl = Slice(bbox, (X, Y, Z), i, samples=samples, orient=orient)
            sl_final = util.overlay_slice(sl, options, window_vals,
                                          img, mask, color_img, None, alpha_img)
            if init:
                implots[i].set_data(sl_final)
                # plt.show()
            else:
                implots[i] = axes[i].imshow(sl_final, origin='lower', extent=sl.extent,
                                            interpolation='nearest')
                axes[i].axis('off')
    
    wrap_sections(bbox.center[0], bbox.center[1], bbox.center[2])
    fig.tight_layout()
    # Setup widgets
    slider_x = ipy.FloatSlider(min=bbox.start[0], max=bbox.end[0], value=bbox.center[0], continuous_update=True)
    slider_y = ipy.FloatSlider(min=bbox.start[1], max=bbox.end[1], value=bbox.center[1], continuous_update=True)
    slider_z = ipy.FloatSlider(min=bbox.start[2], max=bbox.end[2], value=bbox.center[2], continuous_update=True)
    widgets = ipy.interactive(wrap_sections, X=slider_x, Y=slider_y, Z=slider_z)

    # Now do some manual layout
    hbox = ipy.HBox(widgets.children[0:3]) # Set the sliders to horizontal layout
    vbox = ipy.VBox((hbox, widgets.children[3]))
    # iplot.widget.children[-1].layout.height = '350px'
    return vbox