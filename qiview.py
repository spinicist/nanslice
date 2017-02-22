"""qiview.py --- Simple Viewer for 'dual-coded' neuroimaging overlays

Adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

Copyright (C) 2017 Tobias Wood

This code is subject to the terms of the Mozilla Public License. A copy can be
found in the root directory of the project.
"""

import sys
import argparse
import qicommon
import numpy as np
import nibabel as nib
import matplotlib
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5 import QtCore, QtWidgets
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

PROG_NAME = 'QIView'
PROG_VERSION = "0.1"

class QICanvas(FigureCanvas):
    """Canvas to draw slices in."""

    def __init__(self, args, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='k')

        gs1 = GridSpec(1, 3)
        gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
        gs2 = GridSpec(1, 1)
        gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.16, wspace=0.1, hspace=0.1)
        self.axes = []
        for i in range(3):
            self.axes.append(self.fig.add_subplot(gs1[i], facecolor='black'))
        self.cbar_axis = self.fig.add_subplot(gs2[0], facecolor='black')

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.mpl_connect(self, 'button_press_event', self.handle_mouse_event)
        FigureCanvas.mpl_connect(self, 'motion_notify_event', self.handle_mouse_event)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.img_mask = nib.load(args.base_image)
        self.img_base = nib.load(args.mask_image)
        self.img_color = nib.load(args.color_image)
        self.img_alpha = nib.load(args.alpha_image)

        self.corners = qicommon.findCorners(self.img_mask)
        self.cursor = (self.corners[0] + self.corners[1]) / 2
        self.base_window = np.percentile(self.img_base.get_data(), args.window)
        self.args = args
        self.update_figure()

    def update_figure(self):
        directions = ('x', 'y', 'z')
        for i in range(3):
            (this_slice, sl_extent) = qicommon.setupSlice(self.corners[0], self.corners[1], directions[i], 
                                                        self.cursor[i], 128, absolute=True)
            sl_img_mask = qicommon.sampleSlice(self.img_mask, this_slice, order=1)
            sl_base = qicommon.applyCM(qicommon.sampleSlice(self.img_base, this_slice),
                                       'gray', self.base_window)
            sl_color = qicommon.applyCM(qicommon.sampleSlice(self.img_color, this_slice)*self.args.color_scale,
                                        self.args.color_map, self.args.color_lims)
            sl_alpha = qicommon.scaleAlpha(qicommon.sampleSlice(self.img_alpha, this_slice), self.args.alpha_lims)
            sl_blend = qicommon.mask(qicommon.blend(sl_base, sl_color, sl_alpha), sl_img_mask)
            self.axes[i].cla()
            self.axes[i].imshow(sl_blend, origin='lower', extent=sl_extent, interpolation='hanning')
            self.axes[i].contour(sl_alpha, (self.args.contour,), origin='lower', extent=sl_extent)
            self.axes[i].axis('off')
            self.axes[i].axis('image')
        
        # Do these individually now because I'm not clever enough to set them in the loop
        self.axes[0].axhline(y=self.cursor[1], color='g')
        self.axes[0].axvline(x=self.cursor[2], color='g')
        self.axes[1].axhline(y=self.cursor[2], color='g')
        self.axes[1].axvline(x=self.cursor[0], color='g')
        self.axes[2].axhline(y=self.cursor[1], color='g')
        self.axes[2].axvline(x=self.cursor[0], color='g')

        qicommon.alphabar(self.cbar_axis, 
                          self.args.color_map, self.args.color_lims, self.args.color_label,
                          self.args.alpha_lims, self.args.alpha_label)
        self.draw()

    def handle_mouse_event(self, event):
        if event.button == 1:
            if event.inaxes == self.axes[0]:
                self.cursor[1] = event.ydata
                self.cursor[2] = event.xdata
                self.update_figure()
            elif event.inaxes == self.axes[1]:
                self.cursor[0] = event.xdata
                self.cursor[2] = event.ydata
                self.update_figure()
            elif event.inaxes == self.axes[2]:
                self.cursor[0] = event.xdata
                self.cursor[1] = event.ydata
                self.update_figure()
            color_val = qicommon.samplePoint(self.img_color, self.cursor)
            alpha_val = qicommon.samplePoint(self.img_alpha, self.cursor)
            msg = "Cursor: " + str(self.cursor) + " Value: " + str(color_val[0]) + " Alpha: " + str(alpha_val[0])
            # Parent of this is the layout, call parent again to get the main window
            self.parent().parent().statusBar().showMessage(msg)

class ApplicationWindow(QtWidgets.QMainWindow):
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

# pylint insists anything at module level is a constant, so disable the stupidity
# pylint: disable=C0103
parser = argparse.ArgumentParser(description='Dual-coding viewer.')
parser.add_argument('base_image',help='Base (structural image)',type=str)
parser.add_argument('mask_image',help='Mask image',type=str)
parser.add_argument('color_image',help='Image for color-coding of overlay',type=str)
parser.add_argument('alpha_image',help='Image for transparency-coding of overlay',type=str)
parser.add_argument('--window', nargs=2, default=(1,99), help='Specify base image window (in percentiles)')
parser.add_argument('--alpha_lims', nargs=2,default=(0.5,1.0), help='Alpha/transparency window, default=0.5 1.0')
parser.add_argument('--alpha_label', type=str, default='1-p', help='Label for alpha/transparency axis')
parser.add_argument('--contour',help='Specify value for alpha image contour, default=0.95',type=float,default=0.95)
parser.add_argument('--color_lims', type=float, nargs=2, default=(-1,1), help='Colormap window, default=-1 1')
parser.add_argument('--color_scale', type=float, default=1, help='Multiply color image by value, default=1')
parser.add_argument('--color_map', type=str, default='RdYlBu_r', help='Colormap to use from Matplotlib, default = RdYlBu_r')
parser.add_argument('--color_label', type=str, default='% Change', help='Label for color axis')
args = parser.parse_args()
application = QtWidgets.QApplication(sys.argv)
window = ApplicationWindow(args)
window.setWindowTitle("%s" % PROG_NAME)
window.show()
sys.exit(application.exec_())
