# qiview.py --- Simple Viewer for 'dual-coded' neuroimaging overlays
#
# Adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
#
# Copyright (C) 2017 Tobias Wood

import sys
import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import nibabel as nib
import qiplot

prog_name = 'QIView'
prog_version = "0.1"


class QICanvas(FigureCanvas):
    """Canvas to draw slices in."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        print('Loading')
        mask = nib.load('/Users/Tobias/Data/MATRICS/mouse/mask.nii')
        template = nib.load('/Users/Tobias/Data/MATRICS/mouse/stdWarped.nii.gz')
        labels = nib.load('/Users/Tobias/Data/MATRICS/mouse/c57_fixed_labels_resized.nii')
        Tstat = nib.load('/Users/Tobias/Data/MATRICS/mouse/rlog_jacobian_tstat1.nii.gz')
        pstat = nib.load('/Users/Tobias/Data/MATRICS/mouse/rlog_jacobian_vox_p_fstat1.nii')
        print('Setup')
        (tmin, tmax) = np.percentile(template.get_data(), (1,99))
        (corner1, corner2) = qiplot.findCorners(mask)


        cmap = 'RdYlBu_r'
        print('Setup slice')
        (sl, ext) = qiplot.setupSlice(corner1, corner2, 'y', 0.5, 128)
        print('Sample')
        sl_mask = qiplot.sampleSlice(mask, sl, order=1)
        sl_pstat = qiplot.scaleAlpha(qiplot.sampleSlice(pstat, sl), (0.5, 1.0))
        sl_template = qiplot.applyCM(qiplot.sampleSlice(template, sl), 'gray', (tmin, tmax))
        sl_Tstat = qiplot.applyCM(qiplot.sampleSlice(Tstat, sl), cmap, (-4, 4))
        sl_lbls = qiplot.sampleSlice(labels, sl, order=0)
        print('Blend')
        sl_blend = qiplot.mask(qiplot.blend(sl_template, sl_Tstat, sl_pstat), sl_mask)
        print('Plot')
        self.axes.imshow(sl_blend, origin='lower', extent=ext, interpolation='hanning')
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
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
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)
        qc = QICanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(qc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("QIView", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
"""QIView
Copyright 2017 Tobias Wood

A simple viewer for dual-coded overlays.

With thanks to http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html"""
                                )


qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % prog_name)
aw.show()
sys.exit(qApp.exec_())
#qApp.exec_()