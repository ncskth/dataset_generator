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


def camera_pose(radian_y, radian_z, radius, height):
    """
    Calculates poses in (px, py, pz, ox, oy, oz) where ``p`` stands for
    position and ``o`` stands for eulex (xyz) angle.
    """
    x = math.cos(radian_z + math.pi) * radius
    y = math.sin(radian_z + math.pi) * radius
    return f"{x} {y} {height} 0 {radian_y} {radian_z}"


def random_camera_poses(distance: float = None, radius: float = 1.5):
    # Set horizontal panning
    if distance is None:
        distance = random.uniform(1 / 4 * math.pi, math.pi)
    start_z = random.uniform(-math.pi * 2, math.pi * 2)
    end_z = start_z + distance
    # Set vertical panning
    start_y = random.uniform(0, 1)
    y_delta = random.uniform(-0.4, 0.5)
    end_y = start_y + y_delta
    # end_y = start_y
    # Set height difference: [1.1, 2.4]
    start_height = random.uniform(1.3, 1.8)
    height_delta = random.uniform(-0.2, 0.6)
    end_height = start_height + height_delta
    # Generate poses
    return camera_pose(start_y, start_z, radius, start_height), camera_pose(
        end_y, end_z, radius, end_height
    )
