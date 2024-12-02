from jsonargparse import CLI
from sensor_msgs.msg import Image, CameraInfo
import cv_bridge
import numpy as np
import open3d as o3d
import cv2
import yaml
from collections import deque
import rospy
import rosbag
import tqdm

bridge = cv_bridge.CvBridge()

def load_yaml(filepath):
    with open(filepath, 'r') as fp:
        data = yaml.load(fp, yaml.UnsafeLoader)
    return data


def make_o3d_pcd(bgr_image, depth_image, K, P):
    height, width = depth_image.shape
    P = np.linalg.inv(P)
    camera_model = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = o3d.geometry.Image(rgb_image)
    depth_image = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_model, P)
    return pcd


class BagLoader:
    def __init__(self, bag_filepath, window_ms=100.0, stride_ms=50.0, execlude_tf_static=False, batch_output=False, loop=False) -> None:
        self.bag_filepath = bag_filepath
        self.loop = loop
        self.batch_output = batch_output
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.q = deque()
        self.window_duration = rospy.Duration(secs=window_ms/1000.0)
        self.stride_duration = rospy.Duration(secs=stride_ms/1000.0)
        self.execlude_tf_static = execlude_tf_static
        self.start = None
        self.end = None
        self.seq = 0
        self.tf_static_data = []
        self.bag = rosbag.Bag(self.bag_filepath, 'r')
        self.info = self.bag.get_type_and_topic_info()[1]
        self.bag_topics = set(self.info.keys())
        self.bag.close()
        self.bag = None

    def extract_tf_static(self):
        bag = rosbag.Bag(self.bag_filepath, 'r')
        for topic, msg, ts in bag.read_messages('/tf_static'):
            self.tf_static_data.append((topic, msg, ts))

        bag.close()
        return self.tf_static_data

    def _sliding_window_bag_gen(self):
        bag = rosbag.Bag(self.bag_filepath, 'r')

        self.q = deque()
        self.start = None
        self.end = None
        self.seq = 0
        self.tf_static_data = []

        for topic, msg, ts in bag.read_messages():
            if self.start is None:
                self.start = ts
                self.end = self.start + self.window_duration

            if topic == '/tf_static':
                self.tf_static_data.append((topic, msg, ts))
                if self.execlude_tf_static:
                    continue

            self.clear_past()
            if ts >= self.start and ts <= self.end:
                self.q.append((ts, (topic, msg, ts)))
            elif ts > self.end:
                yield [a[1] for a in self.q]
                self.seq += 1
                self.start = self.start + self.stride_duration
                self.end = self.end + self.stride_duration
                self.clear_past()
                self.q.append((ts, (topic, msg, ts)))

        yield [a[1] for a in self.q]
        self.seq += 1

    def clear_past(self):
        while len(self.q) > 0 and self.q[0][0] < self.start:
            self.q.popleft()

    def _update_headers(self, nearest):
        self.missing_topics = set()
        self.offset_duration = {}
        batch = []
        for topic in self.bag_topics:
            if nearest[topic]['offset_duration'] is None:
                self.missing_topics.add(topic)
                self.offset_duration[topic] = None
            else:
                self.offset_duration[topic] = nearest[topic]['offset_duration']

        for topic in nearest:
            if topic in self.missing_topics:
                continue
            _, msg, ts = nearest[topic]['data']
            modified_ts = self.start + self.window_duration/2
            if hasattr(msg, 'header'):
                msg.header.stamp = modified_ts
                msg.header.seq = self.seq
            if hasattr(msg, 'transforms'):
                for transform in msg.transforms:
                    if hasattr(transform, 'header'):
                        transform.header.stamp = modified_ts
                        transform.header.seq = self.seq
            if self.batch_output == False:
                yield topic, msg, modified_ts
            else:
                batch.append((topic, msg, modified_ts))

        if self.batch_output:
            yield batch

    def synched_generator(self, batch_output=False):
        self.batch_output = batch_output
        while True:
            for window_batch in tqdm.tqdm(self._sliding_window_bag_gen()):
                nearest = {topic: {'offset_duration': None, 'data': ()}
                           for topic in self.bag_topics}
                for (topic, msg, ts) in window_batch:
                    offset_duration = abs(
                        ts - self.start - self.window_duration/2)
                    if nearest[topic]['offset_duration'] is None or nearest[topic]['offset_duration'] > offset_duration:
                        nearest[topic]['offset_duration'] = offset_duration
                        nearest[topic]['data'] = (topic, msg, ts)
                yield from self._update_headers(nearest)

            if self.loop == False:
                return
            else:
                print("Looping")
    
    def __iter__(self):
        return self.synched_generator(self.batch_output)
    
    def statistics(self):
        missing_count = Counter()
        n = 0

        for batch in self.synched_generator(batch_output=True):
            missing_count.update(self.missing_topics)
            n += 1

        print('total batches', n)
        print(missing_count)




def visualize_o3d(bag_filepath: str ='/media/marsil/Mext4/pphau_recordings/calibration_2023-01-05-13-51-06.bag',
                  window_ms: int = 100,
                  stride_ms: int = 50,
                  calibration_filepath: str = '/media/marsil/Mext4/pphau_recordings/cams_pairwise_0501.yml'
                  ):
    

    bag_loader = BagLoader(bag_filepath=bag_filepath,
                        window_ms=window_ms,
                        stride_ms=stride_ms,
                        batch_output=True)

    camera_info = load_yaml(calibration_filepath)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    point_cloud = None # o3d.geometry.PointCloud()
    # vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    for batch in bag_loader:
        
        bgr_image = None
        depth_image = None
        data = {serial: {
                         'P': np.array(info['extrinsics']) if info['extrinsics'] is not None else None,
                         'K': np.array(info['intrinsics']['K'])} for serial, info in camera_info.items()}
        # msgs_by_topic = {msg[0]: msg[1] for msg in batch}
        for (topic, msg, ts) in batch:
            if topic.startswith('/camera'):
                serial = topic.split('/')[2]
                if topic.endswith('/color/image_raw/compressed'):
                    data[serial]['bgr_image'] = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                elif topic.endswith('/color/image_raw'):
                    data[serial]['bgr_image'] = bridge.imgmsg_to_cv2(msg, 'bgr8')
                elif topic.endswith('/aligned_depth_to_color/image_raw'):
                    data[serial]['depth_image'] = bridge.imgmsg_to_cv2(msg, '16UC1')
        
        pcd = o3d.geometry.PointCloud()
        for serial, info in data.items():
            P = info['P']
            K = info['K']
            if 'bgr_image' in info and 'depth_image' in info and (P is not None):
                p = make_o3d_pcd(bgr_image=info['bgr_image'], 
                                    depth_image=info['depth_image'],
                                    K=K,
                                    P=P)
                # o3d.visualization.draw_geometries([p])
                pcd += p

        if point_cloud is None:
            point_cloud= pcd
            vis.add_geometry(point_cloud)
        else:
            point_cloud.points = pcd.points
            point_cloud.colors = pcd.colors
            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()
        # if keyboard.is_pressed('q'):
        #     break

if __name__ == '__main__':
    CLI([visualize_o3d])