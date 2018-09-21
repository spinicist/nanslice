#!/usr/bin/env python
"""nanviewer.py

A simple viewer to demonstrate 'dual-coded' neuroimaging overlays. This is installed by PIP
as ``nanviewer``.

Dual-coding is described here https://www.cell.com/neuron/fulltext/S0896-6273(12)00428-X.
Python code adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html.

The viewer can be started on the command-line by calling:

``nanviewer image.nii.gz``

To add a dual-coded overlay, call:

``nanviewer structural.nii.gz --overlay beta.nii.gz --overlay_alpha pval.nii.gz``

For discussion of other options, please see the :py:mod:`~nanslice.nanslicer`
documentation - the majority of options are the same across both programs.
"""
import sys
import argparse
import numpy as np
import scipy.ndimage.interpolation as ndinterp
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets
from .util import add_common_arguments
from .colorbar import colorbar, alphabar
from .slicer import Slicer, axis_indices, Axis_map
from .layer import Layer, blend_layers

PROG_NAME = 'NaNViewer'
PROG_VERSION = "1.0"

def crosshairs(axis, point, direction, orient, color='g'):
    """
    Helper function to draw crosshairs on an axis
    """
    ind1, ind2 = axis_indices(Axis_map[direction], orient)
    vline = axis.axvline(x=point[ind1], color=color)
    hline = axis.axhline(y=point[ind2], color=color)
    return (vline, hline)

def sample_point(img, point, order=1):
    """
    Helper function to sample an image at a single point (instead of a whole slice)
    """
    scale = np.mat(img.get_affine()[0:3, 0:3]).I
    offset = np.dot(-scale, img.get_affine()[0:3, 3]).T
    s_point = np.dot(scale, point).T + offset[:]
    return ndinterp.map_coordinates(img.get_data().squeeze(), s_point, order=order)


class NaNCanvas(FigureCanvas):
    """
    Canvas to draw slices in
    """

    def __init__(self, args, parent=None, width=5, height=4, dpi=100):
        self.layers = [Layer(args.base_image,
                             cmap=args.base_map,
                             clim=args.base_lims,
                             mask=args.mask,
                             interp_order=args.interp_order),]
        if args.overlay:
            self.layers.append(Layer(args.overlay,
                                     cmap=args.overlay_map,
                                     clim=args.overlay_lim,
                                     mask=args.overlay_mask,
                                     mask_threshold=args.overlay_mask_thresh,
                                     alpha=args.overlay_alpha,
                                     alpha_lim=args.overlay_alpha_lim,
                                     interp_order=args.interp_order))

        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='k')
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.mpl_connect(self, 'button_press_event', self.handle_mouse_event)
        FigureCanvas.mpl_connect(self, 'motion_notify_event', self.handle_mouse_event)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        gs1 = GridSpec(1, 3)
        if args.base_map or args.overlay:
            gs2 = GridSpec(1, 1)
            gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
            gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.15, wspace=0.1, hspace=0.1)
            self.cbar_axis = self.fig.add_subplot(gs2[0], facecolor='black')
            if args.overlay_alpha:
                alphabar(self.cbar_axis, args.overlay_map, args.overlay_lim, args.overlay_label,
                         args.overlay_alpha_lim, args.overlay_alpha_label)
            else:
                if args.base_map:
                    colorbar(self.cbar_axis, self.layers[0].cmap,
                             self.layers[0].clim, args.base_label)
                else:
                    colorbar(self.cbar_axis, self.layers[1].cmap,
                             self.layers[1].clim, args.overlay_label)
        else:
            gs1.update(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
        self.axes = [self.fig.add_subplot(gs, facecolor='black') for gs in gs1]
        self.cursor = self.layers[0].bbox.center
        self.args = args
        self._slices = [None, None, None]
        self._images = [None, None, None]
        self._contours = [None, None, None]
        self._crosshairs = [None, None, None]
        self._first_time = True
        self.directions = ('z', 'x', 'y')
        self.update_figure()

    def update_figure(self, hold=None):
        """
        Updates the three axis views
        """
        #t0 = time.time()
        # Save typing and lookup time
        args = self.args
        bbox = self.layers[0].bbox
        cursor = self.cursor
        directions = self.directions
        # Do these individually now because I'm not clever enough to set them in the loop
        if not self._first_time:
            for crosshair in self._crosshairs:
                # Crosshairs consist of a vline/hline pair
                crosshair[0].remove()
                crosshair[1].remove()
        for i in range(3):
            if i != hold:
                self._slices[i] = Slicer(bbox, cursor[i], directions[i],
                                         args.samples, orient=args.orient)
                sl_final = blend_layers(self.layers, self._slices[i])
                # Draw image
                if self._first_time:
                    self._images[i] = self.axes[i].imshow(sl_final, origin='lower',
                                                          extent=self._slices[i].extent,
                                                          interpolation=self.args.interp)
                    # If these calls go in __init__ then images don't show
                    self.axes[i].axis('off')
                    self.axes[i].axis('image')
                else:
                    self._images[i].set_data(sl_final)

                # Draw contours. For contours remove collection manually
                if self.args.contour:
                    if not self._first_time:
                        for coll in self._contours[i].collections:
                            coll.remove()
                    sl_contour = self.layers[1].get_slice(self._slices[i])
                    self._contours[i] = self.axes[i].contour(sl_contour, levels=self.args.contour,
                                                             colors=args.contour_color, linestyles=args.contour_style,
                                                             linewidths=1.0, origin='lower',
                                                             extent=self._slices[i].extent)
            self._crosshairs[i] = crosshairs(self.axes[i], self.cursor,
                                             directions[i], self.args.orient)
        self._first_time = False
        #print('Update time:', (time.time() - t0)*1000, 'ms')
        self.draw()

    def handle_mouse_event(self, event):
        """
        Updates the slice locations and crosshair
        """
        if event.button == 1:
            for i in range(3):
                if event.inaxes == self.axes[i]:
                    ind1, ind2 = axis_indices(Axis_map[self.directions[i]], self.args.orient)
                    self.cursor[ind1] = event.xdata
                    self.cursor[ind2] = event.ydata
                    self.update_figure(hold=i)
            msg = 'Cursor: ' + str(self.cursor)
            if len(self.layers) > 1:
                color_val = sample_point(self.layers[1].base_image,
                                         self.cursor,
                                         self.args.interp_order)
                msg = msg + ' ' + self.args.color_label + ': ' + str(color_val[0])
                if self.layers[1].alpha_image:
                    alpha_val = sample_point(self.layers[1].alpha_image,
                                             self.cursor,
                                             self.args.interp_order)
                    msg = msg + ' ' + self.args.alpha_label + ': ' + str(alpha_val[0])
            # Parent of this is the layout, call parent again to get the main window
            self.parent().parent().statusBar().showMessage(msg)

class NaNViewWindow(QtWidgets.QMainWindow):
    """
    Main window class for the viewer
    """
    def __init__(self, args):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("NaNView")
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self._file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('&About', self._about)
        self.statusBar().showMessage("NaNViewer", 2000)
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(NaNCanvas(args, self.main_widget, width=5, height=4, dpi=100))

    def _file_quit(self):
        """Callback for quit menu entry"""
        self.close()

    def _about(self):
        """Callback for about menu entry"""
        QtWidgets.QMessageBox.about(self, "About", """NaNViewer
Copyright 2017 Tobias Wood

A simple viewer for dual-coded overlays.

With thanks to http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html""")

def main(args=None):
    """
    Main function.

    Parameters:

    - args -- Command-line arguments. See module documentation or command-line for full list
    """
    parser = argparse.ArgumentParser(description='Dual-coding viewer')
    add_common_arguments(parser)
    args = parser.parse_args()
    application = QtWidgets.QApplication(sys.argv)
    window = NaNViewWindow(args)
    window.setWindowTitle("%s" % PROG_NAME)
    window.show()
    sys.exit(application.exec_())

if __name__ == "__main__":
    main()
