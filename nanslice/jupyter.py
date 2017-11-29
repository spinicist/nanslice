#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from . import util
from .box import Box
from .slice import Slice

def slices(img, ncols=3, nrows=1, axis='z', lims=(0.1, 0.9), img_cmap='gray', img_window=(2, 98), mask=None,
           color_img=None, color_cmap='viridis', color_window=None, color_thresh=None, color_label='',
           alpha_img=None, alpha_window=None, alpha_label='',
           contour_img=None, contour_values=(0.95,), contour_colors=('w',), contour_styles=('--',),
           orient='clin', samples=128):
    # Get some information about the image
    if mask:
        bbox = Box.fromMask(mask)
    else:
        bbox = Box.fromImage(img)

    window_vals = np.nanpercentile(img.get_data(), img_window)
    if color_img and color_window is None:
        color_window = np.nanpercentile(color_img.get_data(), (2, 98))
    if alpha_img and alpha_window is None:
        alpha_window = np.nanpercentile(alpha_img.get_data(), (2, 98))
    
    options = util.Options(interp_order=0, color_map=color_cmap, color_lims=color_window, color_scale=1,
                           color_mask_thresh=color_thresh,
                           alpha_lims=alpha_window)

    ntotal = nrows*ncols
    slice_pos = bbox.start + bbox.diag * np.linspace(lims[0], lims[1], ntotal)[:, np.newaxis]

    gs1 = gridspec.GridSpec(nrows, ncols)
    f = plt.figure(facecolor='black', figsize=(ncols*3, nrows*3))

    for s in range(0, ntotal):
        ax = plt.subplot(gs1[s], facecolor='black')
        sl = Slice(bbox, slice_pos[s, :], axis, samples, orient=orient)
        sl_final = util.overlay_slice(sl, options, window_vals, img, mask, color_img, None, alpha_img)
        ax.imshow(sl_final, origin='lower', extent=sl.extent, interpolation='none')
        ax.axis('off')
        if contour_img:
            sl_contour = sl.sample(contour_img, order=1)
            ax.contour(sl_contour, levels=contour_values, origin='lower', extent=sl.extent,
                       colors=contour_colors, linestyles=contour_styles, linewidths=1)

    if color_img:
        gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
        gs2 = gridspec.GridSpec(1, 1)
        gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.15, wspace=0.1, hspace=0.1)
        axes = plt.subplot(gs2[0], facecolor='black')
        if alpha_img:
            util.alphabar(axes, color_cmap, color_window, color_label, alpha_window, alpha_label)
        else:
            util.colorbar(axes, color_cmap, color_window, color_label)
    else:
        gs1.update(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    plt.close()
    return f

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