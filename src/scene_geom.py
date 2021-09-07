import math
from typing import NamedTuple
import random

from tf.transformations import euler_from_quaternion, quaternion_from_euler


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


def camera_pose(radian_x, radian_z, radius):
    x = math.cos(radian_z + math.pi) * radius
    y = math.sin(radian_z + math.pi) * radius
    # Height: [0.5, 1.0]
    height = 0.5 + random.random()
    return f"{x} {y} {height} {radian_x} 0 {radian_z}"


def random_camera_poses(distance: float = None, radius: float = 2.5):
    if distance is None:
        # Set distance [1/4 pi, pi]
        distance = math.pi / 4 + random.random() * (math.pi - math.pi / 4)
    start_z = random.random() * math.pi * 2
    end_z = start_z + distance if random.random() > 0.5 else start_z - distance
    start_x = random.random() * 0.4 - 0.2
    end_x = random.random() * 0.4 - 0.2
    return camera_pose(start_x, start_z, radius), camera_pose(end_x, end_z, radius)
