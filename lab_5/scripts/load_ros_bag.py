import cv2
import rosbag
import open3d as o3d
from cv_bridge import CvBridge
color_topic = '/camera/color/image_raw'
depth_topic = '/camera/aligned_depth_to_color/image_raw'
color_camera_info_topic = '/camera/color/camera_info'
depth_camera_info_topic = '/camera/aligned_depth_to_color/camera_info'

bridge = CvBridge()
topics = [color_topic, depth_topic, color_camera_info_topic, depth_camera_info_topic]

last = {k: None for k in topics}
frames = {k: [] for k in topics}

def load_sequence(path):
    bag = rosbag.Bag(path)
    frame = 0
    last = {topic: None for topic in topics}
    data = []
    pcd_compute = None
    for topic, msg, ts in bag:
        if topic == color_topic:
            msg = bridge.imgmsg_to_cv2(msg, "passthrough")
            color = cv2.cvtColor(msg, cv2.COLOR_BGR2RGB)
            last[topic] = {'msg': color, 'ts': ts, 'topic': topic}
        elif topic == depth_topic: # aligned depth
            depth = bridge.imgmsg_to_cv2(msg, "16UC1")
            last[depth_topic] = {'msg': depth, 'ts': ts, 'topic': topic}
        else:
            last[topic] = {'msg':msg, 'ts':ts, 'topic':topic}

        if topic != color_topic:
            continue
        if any(last[t] is None for t in topics):
            continue


        data.append(last.copy())
        frame += 1

    ret = []
    for topic in topics:
        p = [d[topic]['msg'] for d in data]
        ret.append(p)

    return data, ret


if __name__ == '__main__':
    data, ret = load_sequence('icp_tracking_oats.bag')
    for i in range(50):
        color_image = data[i][color_topic]['msg']
        depth_image = data[i][depth_topic]['msg']
        camera_info = data[i][color_camera_info_topic]['msg']
        
        color_image_o3d = o3d.geometry.Image(color_image)
        depth_image_o3d = o3d.geometry.Image(depth_image)
        rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_image_o3d,
                                                                        depth=depth_image_o3d,
                                                                        depth_scale=1,
                                                                        depth_trunc=3000,
                                                                        convert_rgb_to_intensity=False
                                                                        )
        # rgbd_image_o3d = o3d.geometry.RGBDImage(color=color_image_o3d, depth=depth_image_o3d)

        cv2.imshow("color", color_image)
        cv2.imshow("depth", depth_image)
        print(camera_info)
        k = cv2.waitKey(100)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()