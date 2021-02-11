import time
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import random
import rospy
import os


# parameter
sequences = 5
object_list = ['hammer_simple', 'adjustable_spanner', 'flathead_screwdriver']



## start Virtual Coach
vc = VirtualCoach(environment='local', storage_username='nrpuser', storage_password='password')

for i in range(sequences):
    # START NEW SEQUENCE RUN

    # launch experiment
    sim = vc.launch_experiment('NRPExp_DVSDatabaseGenerator')

    # initialize ros service
    rospy.wait_for_service('gazebo/spawn_sdf_entity')
    spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_entity', SpawnEntity)

    # add objects at random positions
    for model_name in object_list:

        with open(os.path.expandvars('$HBP/Models/') + model_name + '/model.sdf', 'r') as model:
            sdf = model.read()

        pose = Pose()
        pose.position.x = random.uniform(-0.5,0.5)
        pose.position.y = -random.uniform(0,1)
        pose.position.z = 0.9
        pose.orientation.x = random.uniform(0,1)
        pose.orientation.y = random.uniform(0,1)
        pose.orientation.z = random.uniform(0,1)
        pose.orientation.w = random.uniform(0,1)
        reference_frame = 'world'

        res = spawn_model_srv(model_name, sdf, '', pose, reference_frame)
        rospy.loginfo(res)


    # start simulation
    sim.start()

    # wait until experiment finished
    servers = vc.print_available_servers()
    while(servers[0] == 'No available servers.'):
        servers = vc.print_available_servers()
        print(' == Waiting until experiment is finished == ')
        time.sleep(1)

    time.sleep(5)
