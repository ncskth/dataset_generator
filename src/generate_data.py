import time
import argparse
import logging
import random
import os
from pathlib import Path
import tqdm
import xml.etree.ElementTree as ElementTree
import re
import json

import rospy
from pynrp.virtual_coach import VirtualCoach
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose

from scene_geom import random_position, random_rotation, random_camera_poses


def setup_experiment(configuration_file: Path, duration: float):
    timeout_pattern = re.compile(
        '(<timeout time="simulation">).*(</timeout>)', flags=re.MULTILINE
    )
    with open(configuration_file, "r") as fp:
        content = fp.read()
        replacement = timeout_pattern.sub(rf"\g<1>{duration}\g<2>", content)
    with open(configuration_file, "w") as fp:
        fp.write(replacement)


def setup_room(sdffile: Path, pose_start: str, pose_end: str, duration: float):
    tree = ElementTree.parse(sdffile)
    root = tree.getroot()
    waypoints = root.findall(".//trajectory/waypoint/pose")
    waypoint_times = root.findall(".//trajectory/waypoint/time")
    waypoints[0].text = pose_start
    waypoints[-1].text = pose_end
    waypoint_times[-1].text = str(duration)
    tree.write(sdffile)


def record_sequence(vc, object_list, experiment, experiment_path, duration):
    # Setup experiment duration
    setup_experiment(experiment_path / "experiment_configuration.exc", args.duration)

    # prepare room
    pose_start, pose_end = random_camera_poses()
    setup_room(experiment_path / "room.sdf", pose_start, pose_end, duration=duration)

    # launch experiment
    sim = vc.launch_experiment(experiment)

    # initialize ros service
    rospy.wait_for_service("gazebo/spawn_sdf_entity")
    spawn_model_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_entity", SpawnEntity)

    # add objects at random positions
    for model_name in object_list:

        with open(
            os.path.expandvars("$HBP/Models/") + model_name + "/model.sdf", "r"
        ) as model:
            sdf = model.read()

        x, y, z = random_position()
        a, b, c, d = random_rotation()
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = a
        pose.orientation.y = b
        pose.orientation.z = c
        pose.orientation.w = d
        reference_frame = "world"

        res = spawn_model_srv(model_name, sdf, "", pose, reference_frame)
        rospy.loginfo(res)

    # start simulation
    sim.start()
    start = time.time()

    # wait until experiment finished
    servers = vc.print_available_servers()
    while servers[0] == "No available servers.":
        seconds = int(time.time() - start)
        logging.info(f"== Waiting ({seconds}) {servers[0]} ==")
        time.sleep(1)
        # Update servers
        servers = vc.print_available_servers()
    logging.info(f"Complete: {servers[0]}")

    time.sleep(2)


def main(args):
    random.seed(args.random_seed)

    # Start Virtual Coach
    vc = VirtualCoach(storage_username="nrpuser", storage_password="password")

    # Import experiment
    response = vc.import_experiment("NRPExp_DVSDatabaseGenerator")
    experiment_dest = json.loads(response.text)['destFolderName']

    # Load experiment
    storage_path = os.getenv("STORAGE_PATH", "~/.opt/nrpStorage")
    experiment_path = Path(storage_path) / experiment_dest
    object_list = ["hammer_simple"]#, "adjustable_spanner", "flathead_screwdriver"]


    for i in tqdm.tqdm(range(args.recordings)):
        record_sequence(vc, object_list, experiment_dest, experiment_path, args.duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Virtual coach data generator that generates and records one or more scenes"
    )
    parser.add_argument(
        "--recordings", type=int, default=1, help="Number of scenes to record"
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration of the simulation in seconds. Defaults to 1",
    )
    args = parser.parse_args()
    main(args)
