#!/usr/bin/env python
"""qiview.py --- Simple Viewer for 'dual-coded' neuroimaging overlays

Adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

Copyright (C) 2017 Tobias Wood

This code is subject to the terms of the Mozilla Public License. A copy can be
found in the root directory of the project.
"""

import sys
import time
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import numpy as np
import nibabel as nib
from .util import common_arguments, overlay_slice, alphabar, colorbar, crosshairs, sample_point
from .box import Box
from .slicer import Slicer, axis_indices, axis_map
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

PROG_NAME = 'QIView'
PROG_VERSION = "0.1"

class QICanvas(FigureCanvas):
    """Canvas to draw slices in."""

    def __init__(self, args, parent=None, width=5, height=4, dpi=100):
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
        gs1.update(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
        self.axes = [self.fig.add_subplot(gs, facecolor='black') for gs in gs1]
        self.img_base = nib.load(args.base_image)
        self.img_mask = None
        self.img_color = None
        self.img_color_mask = None
        self.img_alpha = None
        self.img_contour = None
        if args.mask:
            self.img_mask = nib.load(args.mask)
            self.bbox = Box.fromMask(self.img_mask)
        else:
            self.bbox = Box.fromImage(self.img_base)
        if args.color:
            self.img_color = nib.load(args.color)
            if args.color_mask:
                self.img_color_mask = nib.load(args.color_mask)
            gs2 = GridSpec(1, 1)
            if args.alpha:
                self.img_alpha = nib.load(args.alpha)
                gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
                gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.16, wspace=0.1, hspace=0.1)
                # If the line below goes before the lines above, it doesn't layout correctly
                self.cbar_axis = self.fig.add_subplot(gs2[0], facecolor='black')
                alphabar(self.cbar_axis, args.color_map, args.color_lims, args.color_label,
                                 args.alpha_lims, args.alpha_label, alines = args.contour)
                if args.contour_img:
                    self.img_contour = nib.load(args.contour_img)
                elif args.contour:
                    self.img_contour = self.img_alpha
            else:
                gs1.update(left=0.01, right=0.99, bottom=0.12, top=0.99, wspace=0.01, hspace=0.01)
                gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.12, wspace=0.1, hspace=0.1)
                # If the line below goes before the lines above, it doesn't layout correctly
                self.cbar_axis = self.fig.add_subplot(gs2[0], facecolor='black')
                colorbar(self.cbar_axis, args.color_map, args.color_lims, args.color_label)
        
        self.cursor = self.bbox.center
        self.base_window = np.percentile(self.img_base.get_data(), args.window)
        self.args = args

        self._slices = [None, None, None]
        self._images = [None, None, None]
        self._contours = [None, None, None]
        self._crosshairs = [None, None, None]
        self._first_time = True
        self.directions = ('z', 'x', 'y')
        self.update_figure()

    def update_figure(self, hold=None):
        """Updates the three axis views"""
        #t0 = time.time()
        # Save typing and lookup time
        args = self.args
        bbox = self.bbox
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
                self._slices[i] = Slice(bbox, cursor, directions[i],
                                        args.samples, orient=args.orient)
                
                sl_final = overlay_slice(self._slices[i], args, self.base_window,
                                                 self.img_base, self.img_mask,
                                                 self.img_color, self.img_color_mask,
                                                 self.img_alpha)
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
                if self.img_contour:
                    if not self._first_time:
                        for coll in self._contours[i].collections:
                            coll.remove()
                    sl_contour = self._slices[i].sample(self.img_contour, order=args.interp_order)
                    self._contours[i] = self.axes[i].contour(sl_contour, levels=self.args.contour,
                                                             colors='k',
                                                             linewidths=1.0, origin='lower',
                                                             extent=self._slices[i].extent)
            self._crosshairs[i] = crosshairs(self.axes[i], self.cursor,
                                                  directions[i], self.args.orient)
        self._first_time = False
        #print('Update time:', (time.time() - t0)*1000, 'ms')
        self.draw()

    def handle_mouse_event(self, event):
        if event.button == 1:
            for i in range(3):
                if event.inaxes == self.axes[i]:
                    ind1, ind2 = axis_indices(axis_map[self.directions[i]], self.args.orient)
                    self.cursor[ind1] = event.xdata
                    self.cursor[ind2] = event.ydata
                    self.update_figure(hold=i)
            msg = 'Cursor: ' + str(self.cursor)
            if self.args.color:
                color_val = sample_point(self.img_color, self.cursor, self.args.interp_order)
                msg = msg + ' ' + self.args.color_label + ': ' + str(color_val[0])
                if self.args.alpha:
                    alpha_val = sample_point(self.img_alpha, self.cursor, self.args.interp_order)
                    msg = msg + ' ' + self.args.alpha_label + ': ' + str(alpha_val[0])
            # Parent of this is the layout, call parent again to get the main window
            self.parent().parent().statusBar().showMessage(msg)

class QIViewWindow(QtWidgets.QMainWindow):
    """Main window class"""
    def __init__(self, args):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QVBoxLayout(self.main_widget)
        qicanvas = QICanvas(args, self.main_widget, width=5, height=4, dpi=100)
        layout.addWidget(qicanvas)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("QIView", 2000)

    def file_quit(self):
        self.close()

    def close_event(self, event):
        self.file_quit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About", """QIView
Copyright 2017 Tobias Wood

A simple viewer for dual-coded overlays.

With thanks to http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html""")

def main(args=None):
    args = common_arguments().parse_args()
    application = QtWidgets.QApplication(sys.argv)
    window = QIViewWindow(args)
    window.setWindowTitle("%s" % PROG_NAME)
    window.show()
    sys.exit(application.exec_())

if __name__ == "__main__":
    main()
