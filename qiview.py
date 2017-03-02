"""qiview.py --- Simple Viewer for 'dual-coded' neuroimaging overlays

Adapted from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

Copyright (C) 2017 Tobias Wood

This code is subject to the terms of the Mozilla Public License. A copy can be
found in the root directory of the project.
"""

import sys
import time
import qicommon as qi
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

        self.img_base = nib.load(args.base_image)
        self.img_mask = nib.load(args.mask_image)
        self.img_color = nib.load(args.color_image)
        self.img_alpha = nib.load(args.alpha_image)

        self.corners = qi.find_bbox(self.img_mask)
        self.cursor = (self.corners[0] + self.corners[1]) / 2
        self.base_window = np.percentile(self.img_base.get_data(), args.window)
        self.args = args

        qi.alphabar(self.cbar_axis,
                    self.args.color_map, self.args.color_lims, self.args.color_label,
                    self.args.alpha_lims, self.args.alpha_label)

        self._slices = [None, None, None]
        self._images = [None, None, None]
        self._contours = [None, None, None]
        self._hlines = [None, None, None]
        self._vlines = [None, None, None]
        self._first_time = True
        self.update_figure()

    def update_figure(self, hold=None):
        """Updates the three axis views"""
        #t0 = time.time()
        directions = ('x', 'y', 'z')
        # Do these individually now because I'm not clever enough to set them in the loop
        if not self._first_time:
            for vline in self._vlines:
                vline.remove()
            for hline in self._hlines:
                hline.remove()
        for i in range(3):
            if i != hold:
                self._slices[i] = qi.Slice(self.corners[0], self.corners[1], directions[i],
                                           self.cursor[i], self.args.samples, absolute=True)
                sl_mask = qi.sample_slice(self.img_mask, self._slices[i], self.args.interp_order)
                sl_base = qi.apply_color(qi.sample_slice(self.img_base,
                                                         self._slices[i],
                                                         self.args.interp_order),
                                         'gray', self.base_window)
                sl_color = qi.apply_color(qi.sample_slice(self.img_color,
                                                          self._slices[i],
                                                          self.args.interp_order)*self.args.color_scale,
                                          self.args.color_map,
                                          self.args.color_lims)
                sl_alpha = qi.scale_clip(qi.sample_slice(self.img_alpha,
                                                         self._slices[i],
                                                         self.args.interp_order),
                                         self.args.alpha_lims)
                sl_blend = qi.mask_img(qi.blend_imgs(sl_base, sl_color, sl_alpha), sl_mask)

                # Draw image
                if self._first_time:
                    self._images[i] = self.axes[i].imshow(sl_blend, origin='lower',
                                                          extent=self._slices[i].extent,
                                                          interpolation=self.args.interp)
                    # If these calls go in __init__ then images don't show
                    self.axes[i].axis('off')
                    self.axes[i].axis('image')
                else:
                    self._images[i].set_data(sl_blend)

                # Draw contours. For contours remove collection manually
                if not self._first_time:
                    for coll in self._contours[i].collections:
                        coll.remove()
                self._contours[i] = self.axes[i].contour(sl_alpha, (self.args.contour,),
                                                         origin='lower',
                                                         extent=self._slices[i].extent)
            self._vlines[i] = self.axes[i].axvline(x=self.cursor[(i+1)%3], color='g')
            self._hlines[i] = self.axes[i].axhline(y=self.cursor[(i+2)%3], color='g')
        self._first_time = False
        #print('Update time:', (time.time() - t0)*1000, 'ms')
        self.draw()

    def handle_mouse_event(self, event):
        if event.button == 1:
            for i in range(3):
                if event.inaxes == self.axes[i]:
                    self.cursor[(i+1)%3] = event.xdata
                    self.cursor[(i+2)%3] = event.ydata
                    self.update_figure(hold=i)
            color_val = qi.sample_point(self.img_color, self.cursor)
            alpha_val = qi.sample_point(self.img_alpha, self.cursor)
            msg = "Cursor: " + str(self.cursor) +\
                  " Value: " + str(color_val[0]) +\
                  " Alpha: " + str(alpha_val[0])
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

# pylint insists anything at module level is a constant, so disable the stupidity
# pylint: disable=C0103
args = qi.common_args().parse_args()
application = QtWidgets.QApplication(sys.argv)
window = QIViewWindow(args)
window.setWindowTitle("%s" % PROG_NAME)
window.show()
sys.exit(application.exec_())
