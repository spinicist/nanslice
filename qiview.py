"""qiview.py --- Simple Viewer for 'dual-coded' neuroimaging overlays

Adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

Copyright (C) 2017 Tobias Wood"""

import sys
import qiplot

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

    def onclick(self, event):
        print('button=', event.button, ' x=', event.x, ' y=', event.y, ' xdata=', event.xdata, ' ydata=', event.ydata)

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='k')

        gs1 = GridSpec(1, 3)
        gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
        gs2 = GridSpec(1, 1)
        gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.16, wspace=0.1, hspace=0.1)

        print('Loading')
        img_mask = nib.load('/Users/Tobias/Data/MATRICS/mouse/mask.nii')
        img_base = nib.load('/Users/Tobias/Data/MATRICS/mouse/stdWarped.nii.gz')
        img_color = nib.load('/Users/Tobias/Data/MATRICS/mouse/rlog_jacobian_tstat1.nii.gz')
        img_alpha = nib.load('/Users/Tobias/Data/MATRICS/mouse/rlog_jacobian_vox_p_fstat1.nii')
        print('Setup')
        (tmin, tmax) = np.percentile(img_base.get_data(), (1, 99))
        (corner1, corner2) = qiplot.findCorners(img_mask)

        cmap = 'RdYlBu_r'
        print('Setup slice')
        slices = []
        self.axes = []
        for direction in ('x', 'y', 'z'):
            (temp_sl, ext) = qiplot.setupSlice(corner1, corner2, direction, 0.5, 128)
            slices.append(temp_sl)

        for i in range(3):
            sl_img_mask = qiplot.sampleSlice(img_mask, slices[i], order=1)
            sl_base = qiplot.applyCM(qiplot.sampleSlice(img_base, slices[i]),
                                     'gray', (tmin, tmax))
            sl_color = qiplot.applyCM(qiplot.sampleSlice(img_color, slices[i]), cmap, (-4, 4))
            sl_alpha = qiplot.scaleAlpha(qiplot.sampleSlice(img_alpha, slices[i]), (0.5, 1.0))
            sl_blend = qiplot.mask(qiplot.blend(sl_base, sl_color, sl_alpha), sl_img_mask)
            axes = fig.add_subplot(gs1[i], facecolor='black')
            axes.imshow(sl_blend, origin='lower', extent=ext, interpolation='hanning')
            axes.axis('off')
            axes.axis('image')
            self.axes.append(axes)
        self.cbar_axis = fig.add_subplot(gs2[0], facecolor='black')
        qiplot.alphabar(self.cbar_axis, cmap, (-4, 4), 'T-Stat', (0.5, 1.0), '1 - p')
        FigureCanvas.__init__(self, fig)
        FigureCanvas.mpl_connect(self, 'button_press_event', self.onclick)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

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
        qicanvas = QICanvas(self.main_widget, width=5, height=4, dpi=100)
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
