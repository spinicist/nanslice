from os import path, getcwd
from .layer import Layer, blend_layers
from .slice_func import scale_clip
from .slicer import Slicer, Axis_map
from .box import Box
from .colorbar import colorbar, alphabar
from .util import add_common_arguments
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec, isdefined


class SlicesInputSpec(BaseInterfaceInputSpec):
    out_file = traits.File(mandatory=True, desc='Output slice image')
    base_file = traits.File(exists=True, mandatory=True,
                            desc='Image file to slice')
    base_map = traits.String('gist_gray', usedefault=True,
                             desc='Color map for base image')
    base_window = traits.Tuple(
        minlen=2, maxlen=2, desc='Window for base image')
    base_scale = traits.Float(1.0, usedefault=True,
                              desc='Scaling factor for base image')
    base_label = traits.String(
        '', usedefault=True, desc='Label for base image color-bar')

    mask_file = traits.File(None, usedefault=True,
                            mandatory=True, desc='Mask file')

    slice_axis = traits.String('z', usedefault=True, desc='Axis to slice')
    slice_lims = traits.Tuple((0.1, 0.9), minlen=2, maxlen=2, usedefault=True,
                              desc='Limits of axis to slice, fraction (low, high) default (0.1, 0.9)')
    slice_layout = traits.Tuple(
        (3, 6), minlen=2, maxlen=2, usedefault=True, desc='Slices layout (rows, cols)')
    volume = traits.Int(0, usedefault=True,
                        desc='Volume for slicing in multi-volume file')
    figsize = traits.Tuple(minlen=2, maxlen=2, desc='Output figure size')
    preclinical = traits.Bool(False, usedefault=True,
                              desc='Data is pre-clinical, fix orientation')
    transpose = traits.Bool(False, usedefault=True,
                            desc='Transpose slice layout')
    bar_pos = traits.String('bottom', usedefault=True,
                            desc='Color/Alphabar position')


class SlicesOutputSpec(TraitedSpec):
    out_file = traits.File(desc="Slice image")


class Slices(BaseInterface):
    input_spec = SlicesInputSpec
    output_spec = SlicesOutputSpec

    def _run_interface(self, runtime):
        inputs = self.inputs
        print('Loading base image: ', inputs.base_file)
        if isdefined(inputs.base_window):
            base_window = inputs.base_window
        else:
            base_window = None
        layers = [Layer(inputs.base_file, mask=inputs.mask_file,
                        cmap=inputs.base_map, clim=base_window, scale=inputs.base_scale,
                        volume=inputs.volume), ]

        bbox = layers[0].bbox
        slice_axis = Axis_map[inputs.slice_axis]

        slice_total = inputs.slice_layout[0]*inputs.slice_layout[1]
        slice_pos = bbox.start[slice_axis] + bbox.diag[slice_axis] * \
            np.linspace(inputs.slice_lims[0],
                        inputs.slice_lims[1], slice_total)
        slice_axis = [slice_axis] * slice_total

        if inputs.preclinical:
            origin = 'upper'
            orient = 'preclin'
        else:
            origin = 'lower'
            orient = 'clin'

        gs1 = gridspec.GridSpec(*inputs.slice_layout)
        if isdefined(inputs.figsize):
            f = plt.figure(facecolor='black', figsize=inputs.figsize)
        else:
            f = plt.figure(facecolor='black', figsize=(
                3*inputs.slice_layout[0], 3*inputs.slice_layout[1] + 1))

        for s in range(0, slice_total):
            if inputs.transpose:
                col, row = divmod(s, inputs.slice_layout[0])
            else:
                row, col = divmod(s, inputs.slice_layout[1])
            ax = plt.subplot(gs1[row, col], facecolor='black')

            slcr = Slicer(bbox, slice_pos[s],
                          slice_axis[s], 256, orient=orient)
            sl_final = blend_layers(layers, slcr)
            ax.imshow(sl_final, origin=origin, extent=slcr.extent)
            ax.axis('off')

        if isdefined(inputs.base_label):
            print('*** Adding colorbar')
            if inputs.bar_pos == 'bottom':
                gs1.update(left=0.01, right=0.99, bottom=(0.4 / (1 + inputs.slice_layout[0])),
                           top=0.99, wspace=0.01, hspace=0.01)
                gs2 = gridspec.GridSpec(1, 1)
                gs2.update(left=0.08, right=0.92, bottom=(0.2 / (1 + inputs.slice_layout[1])),
                           top=(0.38 / (1 + inputs.slice_layout[0])), wspace=0.1, hspace=0.1)
                orient = 'h'
            else:
                gs1.update(left=0.01, right=0.95, bottom=0.01,
                           top=0.99, wspace=0.01, hspace=0.01)
                gs2 = gridspec.GridSpec(1, 1)
                gs2.update(left=0.97, right=0.99, bottom=0.05,
                           top=0.95, wspace=0.01, hspace=0.01)
                orient = 'v'
            axes = plt.subplot(gs2[0], facecolor='black')
            if inputs.base_map:
                colorbar(axes, layers[0].cmap, layers[0].clim,
                         inputs.base_label, orient=orient)
        else:
            gs1.update(left=0.01, right=0.99, bottom=0.01,
                       top=0.99, wspace=0.01, hspace=0.01)
        print('*** Saving')
        print('Writing file: ', inputs.out_file)
        f.savefig(inputs.out_file, facecolor=f.get_facecolor(),
                  edgecolor='none')
        plt.close(f)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = path.abspath(self.inputs.out_file)
        print('Out file:', outputs['out_file'])
        return outputs
