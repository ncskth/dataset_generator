import argparse
import logging
import pathlib
import re
from typing import Dict, NamedTuple
from multiprocessing import Pool

import tqdm 

import rosbag
import geometry_msgs.msg as geomsg

import cameratransform as ct
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import cv2

from cv_bridge import CvBridge


class Datapoint(NamedTuple):
    timestamp: int
    poses: Dict[str, object]
    image_frame: object
    event_frame: object


def get_tool_poses(bag, model_topic):
    # Extract tools
    tools = {}
    for topic, msg, t in bag.read_messages(topics=[model_topic]):
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
    rotation_corrected = rotation.as_euler('xyz')
    rotation = Rotation.from_euler('xyz', rotation_corrected)
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    return position + rotation.apply(mesh)


def camera_pose_generator(bag, model_topic):
    camera_offset = [0, 0, 1.0]
    for topic, msg, t in bag.read_messages(topics=[model_topic]):
        if t.to_time() == 229.716:
            continue # skip weird timestep
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
        yield (t, np.array(list_pose))


def camera_transform_generator(camera_pose_generator, focal_length):
    for t, (x, y, z, roll, pitch, yaw) in camera_pose_generator:
        (roll, pitch, yaw) = (radians * 180 / np.pi for radians in (roll, pitch, yaw))
        projection = ct.RectilinearProjection(
            focallength_px=focal_length, image=(512, 512)
        )
        orientation = ct.SpatialOrientation(
            pos_x_m=x,
            pos_y_m=y,
            elevation_m=z,
            roll_deg=roll+7.7,
            tilt_deg=90 - pitch,
            heading_deg=90 - yaw,
        )
        yield t, ct.Camera(projection, orientation)


def image_generator(bag, bridge, camera_topic):
    for topic, msg, t in bag.read_messages(topics=[camera_topic]):
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        yield (t, image)


def event_generator(bag, event_topic, resolution=[512, 512]):
    rgb_res = (*resolution, 3)
    for topic, msg, t in bag.read_messages(topics=[event_topic]):
        dvs_img = np.zeros(rgb_res, dtype=np.uint8)
        for event in msg.events:
            dvs_img[event.y][event.x] = (event.polarity*255, 255, 0)
        yield (t, dvs_img)


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


def project_labels(camera, tool_poses, tool_meshes, resolution=(512, 512)):
    images = []
    for tool_class, (tool, pose) in enumerate(tool_poses.items()):
        raw_vertices, raw_triangles = tool_meshes[tool]
        transformed = transform_tool(raw_vertices, pose)
        projection = camera.imageFromSpace(transformed)
        triangles = np.take(projection, raw_triangles, axis=0).astype(int)
        image_class = np.zeros(resolution)
        for mesh in triangles:
            cv2.fillConvexPoly(image_class, mesh, tool_class + 1)
        images.append(image_class)
    return np.stack(images, -1)


def align_generators(*generators, key_fn=lambda x: x[0].to_time()):
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
        yield elements


def process_bag(
    bag, bridge, model_topic, camera_topic, event_topic, resolution=[512, 512]
):
    focal_length = 0.5003983220157445 * 512
    models_path = pathlib.Path("../Models/")
    dae_models = models_path.glob("*/*.dae")
    meshes = {key.stem: get_mesh(key) for key in dae_models}
    poses = get_tool_poses(bag, model_topic)

    camera_poses = camera_pose_generator(bag, model_topic)
    cameras = camera_transform_generator(camera_poses, focal_length)
    images = image_generator(bag, bridge, camera_topic)
    events = event_generator(bag, event_topic)

    i = 0
    for (tc, camera), (tp, camera_pose), (ti, rgb), (te, event) in align_generators(
        cameras, camera_poses, images, events
    ):
        labels = project_labels(camera, poses, meshes)
        #yield rgb, camera, camera_pose, poses, meshes, labels, event, (tc, ti, te)
        yield (rgb, event, labels)

def process_dataset(bagfile):
    model_topic = "/gazebo/model_states"
    camera_topic = "/robot/camera_rgb_00"
    event_topic = "/robot/camera_dvs_00/events"
    bridge = CvBridge()
    bagpath = pathlib.Path(bagfile)
    bag = rosbag.Bag(bagpath)
    outpath = bagpath.parent / ('dataset' + bagpath.stem)
    outpath.mkdir()
    logging.debug(f"Processing {bagpath}")
    try:
        for index, (rgb, events, labels) in enumerate(
            process_bag(bag, bridge, model_topic, camera_topic, event_topic)):
            outfile = outpath / f"{index}.npz"
            np.savez(outfile, images=rgb, events=events, labels=labels)
    except Exception as e:
        logging.error("Exception when processing", e)
        pass # Ignore empty generators

def main(args):
    logging.info(f"Processing {len(args.files)} bagfiles")
    with Pool(10) as p:
        list(tqdm.tqdm(p.imap(process_dataset, args.files)))
    #process_dataset(args.files[0])
    print("Done processing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Postprocess NRP DVS datasets")
    parser.add_argument("files", nargs="+", help="Datasets to parse")
    main(parser.parse_args())
