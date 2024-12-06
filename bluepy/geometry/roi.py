""" Geometry primitives describing regions of interests. """
import json
from abc import ABC, abstractmethod

import numpy as np

from bluepy.exceptions import BluePyError
from bluepy.utils import open_utf8


class ROI(ABC):
    """Abstract class representing region of interest."""

    @property
    @abstractmethod
    def volume(self):
        """ Volume. """

    @property
    @abstractmethod
    def bbox(self):
        """ (lower, upper) bounding box. """

    @abstractmethod
    def contains(self, p):
        """ True iff p is within ROI (returns bool array for array of points). """

    @staticmethod
    def load_json(filepath):
        """ Load region of interest from JSON. """
        with open_utf8(filepath) as f:
            data = json.load(f)
        type_ = data['type']
        if type_ == 'sphere':
            params = data['params']
            return Sphere((params['x'], params['y'], params['z']), params['r'])
        else:
            raise BluePyError(f"Unknown ROI type: {type_}")


class Sphere(ROI):
    """ Sphere as a region of interest. """
    def __init__(self, center, radius, closed=True):
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        self.closed = closed

    @property
    def volume(self):
        return (4.0 / 3.0) * np.pi * np.power(self.radius, 3)

    @property
    def bbox(self):
        return self.center - self.radius, self.center + self.radius

    def contains(self, p):
        pred = np.less_equal if self.closed else np.less
        return pred(np.linalg.norm(p - self.center, axis=-1), self.radius)


class Cube(ROI):
    """ Cube as a region of interest. """
    def __init__(self, center, side, closed=True):
        center = np.array(center)
        self.p0 = center - 0.5 * side
        self.p1 = center + 0.5 * side
        self.closed = closed

    @property
    def volume(self):
        return np.prod(self.p1 - self.p0)

    @property
    def bbox(self):
        return self.p0, self.p1

    def contains(self, p):
        pred = np.less_equal if self.closed else np.less
        return np.all(np.logical_and(pred(self.p0, p), pred(p, self.p1)), axis=-1)
