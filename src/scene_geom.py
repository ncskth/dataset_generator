from typing import NamedTuple

from tf.transformations import euler_from_quaternion, quaternion_from_euler

import random


class Table(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z: float = 0.78


table1 = Table(-0.5, 0.5, -0.25, 0.1)
table2 = Table(0.0, 0.5, -1.55, -0.6)


def random_position():
    t = table1 if random.uniform(-1, 1) > 0 else table2
    return (random.uniform(t.x_min, t.x_max), random.uniform(t.y_min, t.y_max), t.z)


def random_rotation():
    return quaternion_from_euler(0, 0, random.uniform(0, 360))