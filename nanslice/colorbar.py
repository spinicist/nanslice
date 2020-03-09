#!/usr/bin/env python
"""colorbar.py

Functions to create a colorbar and a dual-axis color/alphabar.

Matplotlib has no concept of an "alphabar". In addition, because the standard
matplotlib colormaps and colorbars only work on scalar (single-channel) input
images, and matplotlib does not deal with alpha/transparency correctly, nanslice
images are true-color RGB arrays. Hence we need to roll our own colorbar as well
"""
import numpy as np
from . import slice_func


def colorbar(axes, cm_name, clims, clabel,
             black_backg=True, show_ticks=True, tick_fmt='{:.3g}', orient='h'):
    """
    Plots a colorbar in the specified axes

    Parameters:

    - axes -- matplotlib axes instance to use for plotting
    - cm_name -- Colormap name
    - clims -- The limits for the colormap & bar
    - clabel -- Label to place on the color axis
    - black_bg -- Boolean indicating if the background to this plot is black, and hence white text/borders should be used
    - show_ticks -- Set to false if you don't want ticks on the color axis
    - tick_fmt -- Valid format string for the tick labels
    - orient -- 'v' or 'h' for whether you want a vertical or horizontal colorbar
    """
    steps = 32
    if orient == 'h':
        ext = (clims[0], clims[1], 0, 1)
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[
                        np.newaxis, :], [steps, 1])
    else:
        ext = (0, 1, clims[0], clims[1])
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[
                        :, np.newaxis], [1, steps])
    color = slice_func.colorize(cdata, cm_name, clims)
    axes.imshow(color, origin='lower', interpolation='hanning',
                extent=ext, aspect='auto')
    if black_backg:
        forecolor = 'w'
    else:
        forecolor = 'k'
    if show_ticks:
        ticks = (clims[0], np.sum(clims)/2, clims[1])
        labels = (tick_fmt.format(clims[0]),
                  clabel, tick_fmt.format(clims[1]))
        if orient == 'h':
            axes.set_xticks(ticks)
            axes.set_xticklabels(labels, color=forecolor, fontsize=24)
            axes.set_yticks(())
        else:
            axes.set_yticks(ticks)
            axes.set_yticklabels(labels, color=forecolor,
                                 rotation='vertical', va='center')
            axes.yaxis.tick_right()
            axes.set_xticks(())
    else:
        if orient == 'h':
            axes.set_xticks((np.sum(clims)/2,))
            axes.set_xticklabels((clabel,), color=forecolor)
            axes.set_yticks(())
        else:
            axes.set_yticks((np.sum(clims)/2,))
            axes.set_yticklabels((clabel,), color=forecolor)
            axes.yaxis.tick_right()
            axes.set_xticks(())
    axes.tick_params(axis='both', which='both', length=0)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.yaxis.label.set_color(forecolor)
    axes.xaxis.label.set_color(forecolor)
    axes.axis('on')


def alphabar(axes, cm_name, clims, clabel,
             alims, alabel, alines=None, alines_colors=('k',), alines_styles=('solid',),
             cfmt='{:.3g}', afmt='{:.3g}',
             black_backg=True, orient='h'):
    """
    Plots a 2D 'alphabar' with color and transparency axes in the specified matplotlib axes object

    Parameters:

    - axes -- matplotlib axes instance to use for plotting
    - cm_name -- Colormap name
    - clims -- The limits for the colormap & bar
    - clabel -- Label to place on the color axis
    - alims -- The limits for the transparency axis
    - alabel -- Label to place on the transparency axis
    - alines -- Add lines to indicate transparency values (e.g. p < 0.05). Can be a single number or an iterable
    - alines_colors -- Colors to use for each alpha line. Length must match alines parameter
    - alines_styles -- Line styles to use for each alpha line. Length must match alines parameter
    - cprecision -- Precision of color axis ticks
    - aprecision -- Precision of alpha axis ticks
    - black_bg -- Boolean indicating if the background to this plot is black, and hence white text/borders should be used
    - orient -- 'v' or 'h' for whether you want a vertical or horizontal colorbar
    """
    steps = 32
    if orient == 'h':
        ext = (clims[0], clims[1], alims[0], alims[1])
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[
                        np.newaxis, :], [steps, 1])
        alpha = np.tile(np.linspace(0, 1, steps)[:, np.newaxis], [1, steps])
    else:
        ext = (alims[0], alims[1], clims[0], clims[1])
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[
                        :, np.newaxis], [1, steps])
        alpha = np.tile(np.linspace(0, 1, steps)[np.newaxis, :], [steps, 1])
    color = slice_func.colorize(cdata, cm_name, clims)

    backg = np.ones((steps, steps, 3))
    acmap = slice_func.blend(backg, color, alpha)
    axes.imshow(acmap, origin='lower', interpolation='hanning',
                extent=ext, aspect='auto')

    cticks = (clims[0], np.sum(clims)/2, clims[1])
    clabels = (cfmt.format(clims[0]), clabel, cfmt.format(clims[1]))
    aticks = (alims[0], np.sum(alims)/2, alims[1])
    alabels = (afmt.format(alims[0]), alabel, afmt.format(alims[1]))

    if orient == 'h':
        axes.set_xticks(cticks)
        axes.set_xticklabels(clabels)
        axes.set_yticks(aticks)
        axes.set_yticklabels(alabels, rotation='vertical', va='center')
    else:
        axes.set_xticks(aticks)
        axes.set_xticklabels(alabels)
        axes.set_yticks(cticks)
        axes.set_yticklabels(clabels, rotation='vertical', va='center')

    if alines:
        for pos, color, style in zip(alines, alines_colors, alines_styles):
            if orient == 'h':
                axes.axhline(y=pos, linewidth=1.5,
                             linestyle=style, color=color)
            else:
                axes.axvline(x=pos, linewidth=1.5,
                             linestyle=style, color=color)

    if black_backg:
        axes.spines['bottom'].set_color('w')
        axes.spines['top'].set_color('w')
        axes.spines['right'].set_color('w')
        axes.spines['left'].set_color('w')
        axes.tick_params(axis='x', colors='w')
        axes.tick_params(axis='y', colors='w')
        axes.yaxis.label.set_color('w')
        axes.xaxis.label.set_color('w')
    else:
        axes.spines['bottom'].set_color('k')
        axes.spines['top'].set_color('k')
        axes.spines['right'].set_color('k')
        axes.spines['left'].set_color('k')
        axes.tick_params(axis='x', colors='k')
        axes.tick_params(axis='y', colors='k')
        axes.yaxis.label.set_color('k')
        axes.xaxis.label.set_color('k')
        axes.axis('on')
