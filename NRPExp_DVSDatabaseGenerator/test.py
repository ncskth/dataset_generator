# Imported Python Transfer Function
from dvs_msgs.msg import EventArray
import numpy as np
@nrp.MapRobotPublisher('dvs_rendered', Topic('/dvs_rendered_full', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber("dvs_output", Topic('robot/camera_dvs_00/events', EventArray))
@nrp.Robot2Neuron()
def display_events(t, dvs_output, dvs_rendered):
    event_msg = dvs_output.value
    if event_msg is None:
        return
    rendered_img = np.zeros((512, 512, 3), dtype=np.uint8)
    for event in event_msg.events:
        rendered_img[event.y][event.x] = (event.polarity * 255, 255, 0)
    msg_frame = CvBridge().cv2_to_imgmsg(rendered_img, 'rgb8')
    dvs_rendered.send_message(msg_frame)




      # <sensor name='dvs' type='camera'>
      #   <visualize>1</visualize>
      #   <camera>
      #     <horizontal_fov>1.8</horizontal_fov>
      #     <image>
      #       <width>128</width>
      #       <height>128</height>
      #     </image>
      #     <clip>
      #       <near>0.1</near>
      #       <far>100</far>
      #     </clip>
      #   </camera>
      #   <always_on>1</always_on>
      #   <update_rate>60</update_rate>
      #   <visualize>0</visualize>
      #   <plugin name='camera_controller' filename='libgazebo_dvs_plugin.so'>
      #     <cameraName>dvs_left</cameraName>
      #     <robotNamespace>head</robotNamespace>
      #     <eventThreshold>10</eventThreshold>
      #     <cameraInfoTopicName>camera_info</cameraInfoTopicName>
      #   </plugin>
      #   <pose>0 0 0 1.570796327 3.141592654 -0.55</pose>
      # </sensor>
