import rosbag
import os

rosbag_file = os.path.expandvars('$HOME/database_generator_record_2021-02-11-17-56-26.bag')

bag = rosbag.Bag(rosbag_file)

for topic, msg, t in bag.read_messages(topics=[/robot/camera_dvs_00/events']):
   # process rosbag messages

   rint('\n\n BEGIN Message ====== \n\n')
   # print("Topic name is {}".format(topic))
   # print("Message publishing time is: {}".format(t))
   # print("DVS image dimension is: {} (height) x {} (width)".format(msg.height, msg.width))

   for e in msg.events:
        # process all dvs events in message
        print("Printing event {}".format(e))

    print('\n\n END Message ====== \n\n')

bag.close()
