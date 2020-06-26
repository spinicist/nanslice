#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import ipywidgets as ipy
from . import util
from .slicer import Slicer, Axis_map
from .layer import Layer, blend_layers
from .colorbar import colorbar, alphabar


def three_plane(images, orient='clin', samples=128,
                cbar=None, contour=None,
                interactive=False):
    """
    Draw a standard 3-plane view through the center of the image

    Parameters:

    - images -- Either a filename (in a string or Path object) or a list of Layer objects
    - orient -- 'clin' or 'preclin'
    - sample -- Number of samples, default is 128, higher is better but slower
    - cbar -- Adds a colorbar. Either 'True' or an integer giving the layer the colorbar is from
    - contour -- Value(s) that contours are drawn at. Requires that cbar be set - contours will come from same layer
    - interactive -- Add sliders and make the view interactive
    """


def three_plane(images, orient='clin', samples=128,
                volume=0, cbar=None, contour=None,
                interactive=False, title=None):
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    if isinstance(images, str):
        layers = [Layer(images, volume=volume), ]
    elif isinstance(images, Layer):
        layers = [images, ]
    elif isinstance(images[0], str):
        layers = [Layer(img, volume=volume) for img in images]
    else:
        layers = images
    bbox = layers[0].bbox
    gs1 = gs.GridSpec(1, 3)
    fig = plt.figure(facecolor='black', figsize=(9, 3))
    if cbar:
        if isinstance(cbar, bool):
            cbar = 0
        gs1.update(left=0.01, right=0.88, bottom=0.01,
                   top=0.99, wspace=0.01, hspace=0.01)
        gs2 = gs.GridSpec(1, 1)
        gs2.update(left=0.92, right=0.98, bottom=0.1,
                   top=0.9, wspace=0.1, hspace=0.1)
        cax = fig.add_subplot(gs2[0], facecolor='white')
        clayer = layers[cbar]
        if clayer.alpha_image:
            if contour:
                try:
                    iterator = iter(contour)
                except:
                    contour = (contour,)
            alphabar(cax,
                     clayer.cmap, clayer.clim, clayer.label,
                     clayer.alpha_lim, clayer.alpha_label,
                     alines=contour, black_backg=True, orient='v')
        else:
            colorbar(cax,
                     clayer.cmap, clayer.clim, clayer.label,
                     black_backg=True, orient='v')
    else:
        gs1.update(left=0.01, right=0.99, bottom=0.01,
                   top=0.99, wspace=0.01, hspace=0.01)
    implots = [None, None, None]
    iax = [None, None, None]

    def wrap_sections(pos_x, pos_y, pos_z):
        pos = (pos_x, pos_y, pos_z)
        for i in range(3):
            slcr = Slicer(bbox, pos[i], i, samples=samples, orient=orient)
            blended_slice = blend_layers(layers, slcr)
            if implots[i]:
                implots[i].set_data(blended_slice)
            else:
                iax[i] = fig.add_subplot(gs1[i], facecolor='black')
                implots[i] = iax[i].imshow(
                    blended_slice, origin='lower', extent=slcr.extent, interpolation='nearest')
                iax[i].axis('off')
            if contour:
                sl_contour = layers[cbar].get_alpha(slcr)
                iax[i].contour(sl_contour, levels=contour, origin='lower', extent=slcr.extent,
                               colors='k', linestyles='-', linewidths=1)
    wrap_sections(bbox.center[0], bbox.center[1], bbox.center[2])
    if title:
        fig.suptitle(title, color='white')
    if interactive:
        # Setup widgets
        slider_x = ipy.FloatSlider(min=bbox.start[0], max=bbox.end[0], value=bbox.center[0],
                                   continuous_update=True, description='X:')
        slider_y = ipy.FloatSlider(min=bbox.start[1], max=bbox.end[1], value=bbox.center[1],
                                   continuous_update=True, description='Y:')
        slider_z = ipy.FloatSlider(min=bbox.start[2], max=bbox.end[2], value=bbox.center[2],
                                   continuous_update=True, description='Z:')
        widgets = ipy.interactive(
            wrap_sections, pos_x=slider_x, pos_y=slider_y, pos_z=slider_z)
        # Now do some manual layout
        # Set the sliders to horizontal layout
        hbox = ipy.HBox(widgets.children[0:3])
        vbox = ipy.VBox((hbox, widgets.children[3]))
        # iplot.widget.children[-1].layout.height = '350px'
        return vbox
    else:
        plt.close()
        return fig


def slice_axis(images, nrows=1, ncols=3, slice_axis='z', slice_lims=(0.25, 0.75),
               orient='clin', samples=128,
               cbar=None, contour=None, title=None):
    ntotal = nrows*ncols
    slice_axes = slice_axis * ntotal
    slice_pos = np.linspace(slice_lims[0], slice_lims[1], num=ntotal)
    return slices(images, nrows, ncols, slice_axes, slice_pos, False, orient, samples, cbar, contour, title)


def slices(images, nrows=1, ncols=1, slice_axes=['z', ], slice_pos=[0.5, ], absolute=False,
           orient='clin', samples=128,
           cbar=None, contour=None, title=None):
    if isinstance(images, str):
        layers = [Layer(images), ]
    elif isinstance(images, Layer):
        layers = [images, ]
    elif isinstance(images[0], str):
        layers = [Layer(img) for img in images]
    else:
        layers = images
    plt.ioff()
    bbox = layers[0].bbox
    gs1 = gs.GridSpec(nrows, ncols)
    fig = plt.figure(facecolor='black', figsize=(3*ncols, 3*nrows))
    if cbar:
        if isinstance(cbar, bool):
            cbar = 0
        gs1.update(left=0.01, right=0.88, bottom=0.01,
                   top=0.99, wspace=0.01, hspace=0.01)
        gs2 = gs.GridSpec(1, 1)
        gs2.update(left=0.92, right=0.98, bottom=0.1,
                   top=0.9, wspace=0.1, hspace=0.1)
        cax = fig.add_subplot(gs2[0], facecolor='white')
        clayer = layers[cbar]
        if clayer.alpha_image:
            if contour:
                try:
                    iterator = iter(contour)
                except:
                    contour = (contour,)
            alphabar(cax,
                     clayer.cmap, clayer.clim, clayer.label,
                     clayer.alpha_lim, clayer.alpha_label,
                     alines=contour, black_backg=True, orient='v')
        else:
            colorbar(cax,
                     clayer.cmap, clayer.clim, clayer.label,
                     black_backg=True, orient='v')
    else:
        gs1.update(left=0.01, right=0.99, bottom=0.01,
                   top=0.99, wspace=0.01, hspace=0.01)
    for row in range(nrows):
        for col in range(ncols):
            i = row*ncols + col
            axis = slice_axes[i]
            if absolute:
                pos = slice_pos[i]
            else:
                pos = bbox.start[Axis_map[axis]] + \
                    bbox.diag[Axis_map[axis]]*slice_pos[i]
            slcr = Slicer(bbox, pos, axis, samples=samples, orient=orient)
            blended_slice = blend_layers(layers, slcr)
            iax = fig.add_subplot(gs1[row, col], facecolor='black')
            iax.imshow(blended_slice, origin='lower',
                       extent=slcr.extent, interpolation='nearest')
            iax.axis('off')
            if contour:
                sl_contour = layers[cbar].get_alpha(slcr)
                iax.contour(sl_contour, levels=contour, origin='lower', extent=slcr.extent,
                            colors='k', linestyles='-', linewidths=1)
    if title:
        fig.suptitle(title, color='white')
    plt.close()
    return fig


def timeseries(image, axis='z', orient='clin', clim=None, title=None):
    series = Layer(image, clim=clim)
    plt.ioff()
    bbox = series.bbox
    N = series.matrix[3]
    gs1 = gs.GridSpec(1, N)
    fig = plt.figure(facecolor='black', figsize=(3*N, 3))

    ax_ind = Axis_map[axis]
    slcr = Slicer(bbox, bbox.center[ax_ind],
                  axis, samples=series.matrix[ax_ind], orient=orient)
    for i in range(N):
        series.volume = i
        sl = series.get_color(slcr)
        iax = fig.add_subplot(gs1[i], facecolor='black')
        iax.imshow(sl, origin='lower', extent=slcr.extent,
                   interpolation='nearest')
        iax.axis('off')
    if title:
        plt.suptitle(title, color='w')
    plt.close()
    return fig


def compare(image1, image2, axis='z', orient='clin', samples=128,
            clim=None, title=None):
    layer1 = Layer(image1, interp_order=0, clim=clim)
    layer2 = Layer(image2, interp_order=0, clim=layer1.clim)
    plt.ioff()
    bbox = layer1.bbox
    gs1 = gs.GridSpec(1, 2)
    fig = plt.figure(facecolor='black', figsize=(6, 3))

    gs1.update(left=0.01, right=0.88, bottom=0.01,
               top=0.99, wspace=0.01, hspace=0.01)
    gs2 = gs.GridSpec(1, 1)
    gs2.update(left=0.92, right=0.98, bottom=0.1,
               top=0.9, wspace=0.1, hspace=0.1)
    cax = fig.add_subplot(gs2[0], facecolor='white')
    colorbar(cax,
             layer1.cmap, layer1.clim, layer1.label,
             black_backg=True, orient='v')

    ax_ind = Axis_map[axis]
    slcr = Slicer(bbox, bbox.center[ax_ind],
                  axis, samples=layer1.matrix[ax_ind], orient=orient)
    slice1 = layer1.get_color(slcr)
    slice2 = layer2.get_color(slcr)
    iax = fig.add_subplot(gs1[0], facecolor='black')
    iax.imshow(slice1, origin='lower', extent=slcr.extent,
               interpolation='nearest')
    iax.axis('off')
    iax = fig.add_subplot(gs1[1], facecolor='black')
    iax.imshow(slice2, origin='lower', extent=slcr.extent,
               interpolation='nearest')
    iax.axis('off')
    if title:
        fig.suptitle(title, color='white')
    plt.close()
    return fig
