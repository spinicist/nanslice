"""qiview.py --- Simple Viewer for 'dual-coded' neuroimaging overlays

Adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

Copyright (C) 2017 Tobias Wood

This code is subject to the terms of the Mozilla Public License. A copy can be
found in the root directory of the project.
"""

import sys
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

    def __init__(self, parent=None, width=5, height=4, dpi=100,
                 base_file=None, mask_file=None, color_file=None, alpha_file=None):
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

        if base_file is None or mask_file is None or color_file is None or alpha_file is None:
            raise ValueError('Must specify base, mask, color and alpha images')
        self.img_mask = nib.load(base_file)
        self.img_base = nib.load(mask_file)
        self.img_color = nib.load(color_file)
        self.img_alpha = nib.load(alpha_file)
        (corner1, corner2) = qicommon.findCorners(self.img_mask)
        self.cursor = (corner1 + corner2) / 2
        self.update_figure()

    def update_figure(self):
        (tmin, tmax) = np.percentile(self.img_base.get_data(), (1, 99))
        (corner1, corner2) = qicommon.findCorners(self.img_mask)

        cmap = 'RdYlBu_r'
        directions = ('x', 'y', 'z')
        for i in range(3):
            (this_slice, sl_extent) = qicommon.setupSlice(corner1, corner2, directions[i], 
                                                        self.cursor[i], 128, absolute=True)
            sl_img_mask = qicommon.sampleSlice(self.img_mask, this_slice, order=1)
            sl_base = qicommon.applyCM(qicommon.sampleSlice(self.img_base, this_slice),
                                     'gray', (tmin, tmax))
            sl_color = qicommon.applyCM(qicommon.sampleSlice(self.img_color, this_slice), cmap, (-4, 4))
            sl_alpha = qicommon.scaleAlpha(qicommon.sampleSlice(self.img_alpha, this_slice), (0.5, 1.0))
            sl_blend = qicommon.mask(qicommon.blend(sl_base, sl_color, sl_alpha), sl_img_mask)
            self.axes[i].cla()
            self.axes[i].imshow(sl_blend, origin='lower', extent=sl_extent, interpolation='hanning')
            self.axes[i].contour(sl_alpha, (0.95,), origin='lower', extent=sl_extent)
            self.axes[i].axis('off')
            self.axes[i].axis('image')
        
        # Do these individually now because I'm not clever enough to set them in the loop
        self.axes[0].axhline(y=self.cursor[1], color='g')
        self.axes[0].axvline(x=self.cursor[2], color='g')
        self.axes[1].axhline(y=self.cursor[2], color='g')
        self.axes[1].axvline(x=self.cursor[0], color='g')
        self.axes[2].axhline(y=self.cursor[1], color='g')
        self.axes[2].axvline(x=self.cursor[0], color='g')

        qicommon.alphabar(self.cbar_axis, cmap, (-4, 4), 'T-Stat', (0.5, 1.0), '1 - p')
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
    def __init__(self):
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
        qicanvas = QICanvas(self.main_widget, width=5, height=4, dpi=100,
                            base_file=sys.argv[1],
                            mask_file=sys.argv[2],
                            color_file=sys.argv[3],
                            alpha_file=sys.argv[4])
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
application = QtWidgets.QApplication(sys.argv)
window = ApplicationWindow()
window.setWindowTitle("%s" % PROG_NAME)
window.show()
sys.exit(application.exec_())
