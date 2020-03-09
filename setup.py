"""Setup file to Not Another Neuroimaging Slicer

"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nanslice',
    version='1.0.0',
    description='Scripts to slice and display neuroimages (probably stored in nifti format)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/spinicist/nanslice',
    author='Tobias Wood',
    author_email='tobias@spinicist.org.uk',
    py_modules=['nanslice'],
    install_requires=['matplotlib>=3.2.0',
                      'nibabel>=2.0.0',
                      'numpy>=1.14.2',
                      'scipy>=1.1.0',
                      'colorcet>=2.0.0'],
    python_requires='>=3',
    license='MPL',
    classifiers=['Topic :: Scientific/Engineering :: Visualization',
                 'Programming Language :: Python :: 3',
                 ],
    keywords='neuroimaging nifti',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nanslicer=nanslice.nanslicer:main',
            'nanviewer=nanslice.nanviewer:main',
            'nanscroll=nanslice.nanscroll:main'
        ],
    },
)
