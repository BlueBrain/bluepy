#!/usr/bin/env python
""" bluepy setup """
from setuptools import setup, find_packages

import importlib.util

spec = importlib.util.spec_from_file_location("bluepy.version", "bluepy/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

EXTRA_BBP = [
    'brion>=3.3.0,<4.0',
]

EXTRA_FLATINDEX = [
    'libflatindex>=1.8.2,<2.0',
]

EXTRA_NGV = [
    'tables>=3.4',
]

setup(
    name="bluepy",
    version=VERSION,
    python_requires=">=3.8",
    install_requires=[
        'libsonata>=0.1.7,<1.0.0',
        'voxcell>=3.0.0,<4.0.0',
        'bluepy-configfile>=0.1.20,<1.0.0',
        'numpy>=1.8.0,<2.0.0',
        'h5py>=3.0.1,<4.0.0',
        'cached_property>=1.0',
        'pandas>=1.0.0,!=2.0.0,<3',
        'morph_tool>=2.4.3,<3.0.0',
        'morphio>=3.0.1,<4.0.0',
    ],
    extras_require={
        'all': [
            'matplotlib>=3.0.0,<4.0.0',
        ] + EXTRA_BBP + EXTRA_NGV,
        'bbp': EXTRA_BBP,
        'flatindex': EXTRA_FLATINDEX,
        'ngv': EXTRA_NGV,
    },
    packages=find_packages(),
    author="Eilif Muller, Arseny V. Povolotsky",
    author_email="arseny.povolotsky@epfl.ch",
    description="The Pythonic Blue Brain data access API",
    long_description="The Pythonic Blue Brain data access API",
    long_description_content_type="text/plain",
    license="BBP-internal-confidential",
    keywords=[
        'computational neuroscience',
        'simulation',
        'analysis',
        'visualization',
        'parameters',
        'BlueBrainProject',
    ],
    url="https://bbpteam.epfl.ch/documentation/projects/bluepy",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/BLPY/issues",
        "Source": "https://bbpgitlab.epfl.ch/nse/bluepy.git",
    },
    download_url="https://bbpteam.epfl.ch/repository/devpi/+search?query=name:bluepy",
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: Proprietary',
                 'Operating System :: POSIX',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Utilities',
                 ],
)
