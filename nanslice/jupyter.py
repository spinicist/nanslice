#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import ipywidgets as ipy
from . import util
from .box import Box
from .slicer import Slicer
from .layer import Layer, blend_layers

def slices(img, ncols=3, nrows=1, axis='z', lims=(0.1, 0.9),
           cmap=None, clim=None, label='',
           mask=None, overlays=None,
           contour_img=None, contour_values=(0.95,), contour_colors=('w',), contour_styles=('--',),
           orient='clin', samples=256):
    """Draws a grid of slices through an image"""
    base = Layer(img, cmap=cmap, clim=clim, mask=mask)
    if mask:
        bbox = Box.fromMask(base.mask_image)
    else:
        bbox = Box.fromImage(base.image)
    ntotal = nrows*ncols
    slice_pos = bbox.start + bbox.diag * np.linspace(lims[0], lims[1], ntotal)[:, np.newaxis]

    gs1 = gs.GridSpec(nrows, ncols)
    fig = plt.figure(facecolor='black', figsize=(ncols*3, nrows*3))

    for s in range(0, ntotal):
        axes = plt.subplot(gs1[s], facecolor='black')
        slcr = Slicer(bbox, slice_pos[s, :], axis, samples, orient=orient)
        sl_final = blend_layers([base, overlays], slcr)
        axes.imshow(sl_final, origin='lower', extent=slcr.extent, interpolation='nearest')
        axes.axis('off')
        if contour_img:
            sl_contour = slcr.sample(contour_img, order=1)
            axes.contour(sl_contour, levels=contour_values, origin='lower', extent=slcr.extent,
                         colors=contour_colors, linestyles=contour_styles, linewidths=1)

    if cmap:
        gs1.update(left=0.01, right=0.9, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
        gs2 = gs.GridSpec(1, 1)
        gs2.update(left=0.9, right=0.99, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
        axes = plt.subplot(gs2[0], facecolor='black')
        util.colorbar(axes, base.cmap, base.clim, label, black_backg=True, orient='v')
    fig.tight_layout()
    plt.close()
    return fig

def three_plane(img,
                cmap=None, clim=None, label='',
                mask=None, overlays=None,
                orient='clin', samples=128):
    """Draw a standard 3-plane view through the center of the image"""
    base = Layer(img, cmap=cmap, clim=clim, mask=mask)
    if mask:
        bbox = Box.fromMask(base.mask_image)
    else:
        bbox = Box.fromImage(base.image)
    if cmap:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), facecolor='black')
        util.colorbar(axes[3], base.cmap, base.clim, label, black_backg=True, orient='v')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor='black')
    for i in range(3):
        slcr = Slicer(bbox, bbox.center, i, samples=samples, orient=orient)
        blended_slice = blend_layers([base, overlays], slcr)
        axes[i].imshow(blended_slice, origin='lower', extent=slcr.extent, interpolation='nearest')
        axes[i].axis('off')
    fig.tight_layout()
    plt.close()
    return fig

def three_plane_viewer(img,
                       cmap=None, clim=None, label='',
                       mask=None, overlays=None,
                       orient='clin', samples=128):
    """Standard 3-plane view with sliders to control X, Y, Z"""
    plt.ion()
    base = Layer(img, cmap=cmap, clim=clim, mask=mask)
    if mask:
        bbox = Box.fromMask(base.mask_image)
    else:
        bbox = Box.fromImage(base.image)
    if cmap:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), facecolor='black')
        util.colorbar(axes[3], base.cmap, base.clim, label, black_backg=True, orient='v')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), facecolor='black')
    implots = [None, None, None]
    def wrap_sections(pos_x, pos_y, pos_z):
        for i in range(3):
            slcr = Slicer(bbox, (pos_x, pos_y, pos_z), i, samples=samples, orient=orient)
            blended_slice = blend_layers([base, overlays], slcr)
            if implots[i]:
                implots[i].set_data(blended_slice)
            else:
                implots[i] = axes[i].imshow(blended_slice, origin='lower',
                                            extent=slcr.extent, interpolation='nearest')
                axes[i].axis('off')
                plt.show()
    wrap_sections(bbox.center[0], bbox.center[1], bbox.center[2])
    fig.tight_layout()
    # Setup widgets
    slider_x = ipy.FloatSlider(min=bbox.start[0], max=bbox.end[0], value=bbox.center[0],
                               continuous_update=True, description='X:')
    slider_y = ipy.FloatSlider(min=bbox.start[1], max=bbox.end[1], value=bbox.center[1],
                               continuous_update=True, description='Y:')
    slider_z = ipy.FloatSlider(min=bbox.start[2], max=bbox.end[2], value=bbox.center[2],
                               continuous_update=True, description='Z:')
    widgets = ipy.interactive(wrap_sections, pos_x=slider_x, pos_y=slider_y, pos_z=slider_z)
    # Now do some manual layout
    hbox = ipy.HBox(widgets.children[0:3]) # Set the sliders to horizontal layout
    vbox = ipy.VBox((hbox, widgets.children[3]))
    # iplot.widget.children[-1].layout.height = '350px'
    return vbox
