&quot;=3D"import argparse
import datetime
import logging
import pathlib
import re
from typing import Dict, NamedTuple
from multiprocessing import Pool

import tqdm

import rosbag
import geometry_msgs.msg as geomsg

from sklearn.decomposition import PCA

import cameratransform as ct
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import cv2

from cv_bridge import CvBridge


def get_tool_poses(bag, model_topic):
    # Extract tools
    tools = {}
    for topic, gazebo_msg, t in bag.read_messages(topics=[model_topic]):
        msg = gazebo_msg.gazebo_model_states
        if len(msg.pose) > 1:
            for tool_id, tool_name in enumerate(msg.name):
                if "camera" not in tool_name:
                    tools[tool_name] = msg.pose[tool_id]
            break
    return tools


def transform_tool(mesh, pose):
    orientation = pose.orientation
    orientation_quat = np.array(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    )
    rotation = Rotation.from_quat(orientation_quat)
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    return position + rotation.apply(mesh)


def camera_pose_generator(bag, model_topic):
    camera_offset = [0, 0, 1.0]
    for topic, gazebo_msg, t in bag.read_messages(topics=[model_topic]):
        msg = gazebo_msg.gazebo_model_states
        camera_pose = msg.pose[0]
        orientation = camera_pose.orientation
        quaternion = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        euler = euler_from_quaternion(quaternion)
        list_pose = [
            camera_pose.position.x + camera_offset[0],
            camera_pose.position.y + camera_offset[1],
            camera_pose.position.z + camera_offset[2],
        ]
        list_pose.extend(euler)
        yield (message_to_time(gazebo_msg), np.array(list_pose))


def transform_camera(camera_pose, focal_length, resolution):
    (x, y, z, roll, pitch, yaw) = camera_pose
    (roll, pitch, yaw) = (radians * 180 / np.pi for radians in (roll, pitch, yaw))
    projection = ct.RectilinearProjection(focallength_px=focal_length, image=resolution)
    orientation = ct.SpatialOrientation(
        pos_x_m=x,
        pos_y_m=y,
        elevation_m=z,
        roll_deg=roll,
        tilt_deg=90 - pitch,
        heading_deg=90 - yaw,
    )
    return ct.Camera(projection, orientation)


def image_generator(bag, bridge, camera_topic):
    for topic, msg, t in bag.read_messages(topics=[camera_topic]):
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        yield (message_to_time(msg), image)


def event_generator(bag, event_topic, resolution=[640, 480]):
    # Note: We reverse the resolution to align with row-format HxW
    rgb_res = (*list(reversed(resolution)), 2)
    for topic, msg, t in bag.read_messages(topics=[event_topic]):
        dvs_img = np.zeros(rgb_res, dtype=np.uint8)
        for event in msg.events:
            # Note: we assign HxW to support row-format
            dvs_img[event.y][event.x] = (int(event.polarity), int(not event.polarity))
        yield (message_to_time(msg), dvs_img)


def get_mesh(model_path):
    all_vertices_meshes = np.zeros((0, 3))
    all_triangles_meshes = np.zeros((0, 3))

    with open(model_path) as fp:
        xml = fp.read()
        vertices_info = re.findall(
            r"<float_array.+?mesh-positions-array.+?>(.+?)</float_array>", xml
        )
        transform_info = re.findall(
            r'<matrix sid="transform">(.+?)</matrix>.+?<instance_geometry',
            xml,
            flags=re.DOTALL,
        )  # better way?
        triangles_info = re.findall(
            r"<triangles.+?<p>(.+?)</p>.+?</triangles>", xml, flags=re.DOTALL
        )
        if len(triangles_info) == 0:
            triangles_info = re.findall(
                r"<polylist.+?<p>(.+?)</p>.+?</polylist>", xml, flags=re.DOTALL
            )
        for part_id in range(len(vertices_info)):
            transform_matrix = np.array(
                [float(n) for n in transform_info[part_id].split(" ")]
            ).reshape((4, 4))
            vertices_temp = np.array(
                [float(n) for n in vertices_info[part_id].split(" ")]
            )
            vertices_temp = np.reshape(
                vertices_temp, (int(vertices_temp.shape[0] / 3), 3)
            )
            vertices_temp = np.dot(
                transform_matrix,
                np.c_[vertices_temp, np.ones(vertices_temp.shape[0])].T,
            )[:-1].T
            triangles_temp = np.array(
                [int(n) for n in triangles_info[part_id].split(" ")]
            )[::3]
            triangles_temp = np.reshape(
                triangles_temp, (int(triangles_temp.shape[0] / 3), 3)
            )
            triangles_temp = (
                triangles_temp + all_vertices_meshes.shape[0]
            )  # shift triangle indices
            all_vertices_meshes = np.vstack((all_vertices_meshes, vertices_temp))
            all_triangles_meshes = np.vstack((all_triangles_meshes, triangles_temp))

    return all_vertices_meshes, all_triangles_meshes.astype(int)


def project_labels(camera, tool_poses, tool_meshes, resolution):
    # Note: We reverse the resolution to align with row-format HxW
    image_class = np.zeros(list(reversed(resolution)))
    tool_pose_labels = []
    for tool_class, (tool, pose) in enumerate(tool_poses.items()):
        raw_vertices, raw_triangles, centroid = tool_meshes[tool]

        # Fill in segmentation polygons
        transformed_vertices = transform_tool(raw_vertices, pose)
        projection = camera.imageFromSpace(transformed_vertices)
        triangles = np.take(projection, raw_triangles, axis=0).astype(int)
        for mesh in triangles:
            # Tool class is incremented such that 0 becomes "background"
            cv2.fillConvexPoly(image_class, mesh, tool_class + 1)

        # Project tool pose to camera with PCA
        pca = PCA(n_components=2)
        transformed_coords = pca.fit_transform(transformed_vertices)
        transformed_center = np.mean(transformed_coords, dims=0)
        pca_center = pca.inverse_transform(transformed_center)
        x, y = camera.imageFromSpace(pca_center)
        depth = np.linalg.norm(pca_center - camera.getPos())
        tool_pose_labels.append(np.array(x, y, depth))

    return image_class, tool_pose_labels


def message_to_time(msg):
    def stamp_to_time(stamp):
        minutes = stamp.secs // 60
        seconds = stamp.secs % 60
        ms = stamp.nsecs // 1000
        return datetime.time(minute=minutes, second=seconds, microsecond=ms)

    if isinstance(msg, datetime.time):
        return msg
    if hasattr(msg, "header"):
        return stamp_to_time(msg.header.stamp)
    if hasattr(msg, "gazebo_model_states_header"):
        return stamp_to_time(msg.gazebo_model_states_header.stamp)
    raise ValueError("No timestamp in ", msg)


def align_generators(
    *generators, key_fn=lambda x: message_to_time(x[0]), value_fn=lambda x: x[1]
):
    is_empty = False

    while not is_empty:
        elements = [next(x) for x in generators]
        keys = list(map(key_fn, elements))
        max_key = max(keys)
        for index in range(len(keys)):
            while keys[index] < max_key:
                new_element = next(generators[index])
                new_key = key_fn(new_element)
                elements[index] = new_element
                keys[index] = new_key
        yield zip(keys, map(value_fn, elements))


def process_bag(
    bag,
    bridge,
    model_topic,
    camera_topic,
    event_topic,
    resolution=[640, 480],
    model_path="Models",
):
    focal_length = 0.5003983220157445 * resolution[0]
    models_path = pathlib.Path(model_path)
    assert models_path.exists(), f"Could not find the path to Models: {model_path}"
    dae_models = models_path.glob("*/*.dae")
    meshes = {key.stem: get_mesh(key) for key in dae_models}
    poses = get_tool_poses(bag, model_topic)

    camera_poses = camera_pose_generator(bag, model_topic)
    images = image_generator(bag, bridge, camera_topic)
    events = event_generator(bag, event_topic)

    # Remove first few event from buggy topics
    next(camera_poses)
    next(camera_poses)
    next(images)
    next(images)

    i = 0
    for (tp, camera_pose), (ti, rgb), (te, event) in align_generators(
        camera_poses, images, events
    ):
        camera = transform_camera(camera_pose, focal_length, resolution)
        labels, tool_poses = project_labels(camera, poses, meshes, resolution)
        # yield rgb, camera, camera_poseposes, meshes, labels, event, (tc, ti, te)
        yield (rgb, event, labels, tool_poses)


def process_dataset(bagfile):
    model_topic = "/gazebo_modelstates_with_timestamp"
    camera_topic = "/robot/camera_rgb_00"
    event_topic = "/robot/camera_dvs_00/events"
    bridge = CvBridge()
    bagpath = pathlib.Path(bagfile)
    bag = rosbag.Bag(bagpath)
    logging.debug(f"Processing {bagpath}")
    frames, events, labels, poses = [], [], [], []
    try:
        for index, (rgb, event, label, pose) in enumerate(
            process_bag(bag, bridge, model_topic, camera_topic, event_topic)
        ):
            frames.append(rgb)
            events.append(event)
            labels.append(label)
            poses.append(pose)
    except Exception as e:
        if "StopIteration" in str(e):
            pass  # Ignore empty generators
        else:
            logging.error("Exception when processing", e)

    outpath = bagpath.parent / f"{bagpath.stem}.npz"
    np.savez(
        outpath,
        frames=np.stack(frames),
        events=np.stack(events),
        labels=np.stack(labels),
        poses=np.stack(poses),
    )


def main(args):
    logging.info(f"Processing {len(args.files)} bagfiles")
    with Pool(10) as p:
        list(tqdm.tqdm(p.imap(process_dataset, args.files)))
    logging.info(f"Done processing {len(args.files)} bagfiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Postprocess NRP DVS datasets")
    parser.add_argument("files", nargs="+", help="Datasets to parse")
    main(parser.parse_args())
