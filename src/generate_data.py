import time
from pynrp.virtual_coach import VirtualCoach
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import argparse
import random
import rospy
import os


from scene_geom import random_position, random_rotation


def record_sequence(vc, object_list):
    # launch experiment
    sim = vc.launch_experiment("NRPExp_DVSDatabaseGenerator_0")

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
        # servers = vc.print_available_servers()
        seconds = int(time.time() - start)
        print(f" == Waiting until experiment is finished ({ctime}) == ")
        time.sleep(1)

    time.sleep(2)


def main(args):
    random.seed(args.random_seed)

    object_list = ["hammer_simple", "adjustable_spanner", "flathead_screwdriver"]

    ## start Virtual Coach
    vc = VirtualCoach(storage_username="nrpuser", storage_password="password")

    for i in range(args.sequences):
        record_sequence(vc, object_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Virtual coach data generator that generates and records one or more scenes"
    )
    parser.add_argument(
        "--sequences", type=int, default=1, help="Number of times a scene is recorded"
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    main(args)