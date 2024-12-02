import cv2
import rosbag
import cv_bridge
import numpy as np
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation
from sensor_msgs.msg import Image, CameraInfo
bridge = cv_bridge.CvBridge()


def topic2key(topic):
    return tuple(topic.split('/')[-3:])






def calibrate_extrinsics_pnp(images, n, mtx):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 9)  # (6, 9)
    # CHECKERBOARD = (9, 6)  # (6, 9)

    # criteria = (10, 30000000, 0.0000001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * .04
    # images = [c1_colorimage, c2_colorimage]
    # for _ in range(1):
    for p in range(n):
        img = images[p]
        # cv2.imshow('',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            # objpoints.append(objp)
            # r, g, b = cv2.split(img)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)


    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """

    dist = np.array([[0.00000], [0.00000], [0.00000], [0.00000], [0.00000 ]])
    # print(np.array(imgpoints).shape)



    rvecs = {}
    tvecs = {}
    ret = {}

    npobjpoints = np.array(objpoints)
    npimgpoints = np.array(imgpoints)

    # ret, mtx1, dist1, rvecs, tvecs = cv2.calibrateCamera(objectPoints=npobjpoints,imagePoints= npimgpoints,
    #                                                      imageSize=gray.shape[::-1], cameraMatrix=mtx, distCoeffs=dist, flags=None, criteria=criteria)


    ret[0], rvecs[0], tvecs[0] = cv2.solvePnP(objectPoints=objp, imagePoints=npimgpoints[0],
                                              cameraMatrix=mtx, distCoeffs=dist, useExtrinsicGuess=False, flags=0)

    for i in range(1,n):
        ret[i], rvecs[i], tvecs[i] = cv2.solvePnP(objectPoints=objp, imagePoints=npimgpoints[i],
                                               cameraMatrix=mtx, distCoeffs=dist, rvec= rvecs[i-1], tvec= tvecs[i-1],useExtrinsicGuess=True, flags=0)


    print(ret)
    rot_cam = {}
    transf= {}
    p = np.array([0, 0, 0, 1], dtype=np.double)

    for i in range(n):


        rot_cam[i] = cv2.Rodrigues(rvecs[i])
        transf[i] = np.concatenate((rot_cam[i][0], tvecs[i]), axis=1)
        transf[i] = np.vstack((transf[i], p))
        # print(f"Trans Cam{i}: \n", transf[i])
    return transf[n-1]



def calibrate_exrinsics(images, K, square_size, height, width):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    # objp = objp * square_size
    num_cameras, num_images = len(images), len(images[0])
    output = []
    for camera_index in range(num_cameras):
        objpoints = []
        imgpoints = []
        cameraMatrix = K[camera_index].copy()
        print(cameraMatrix)
        for  image_index in range(num_images):
            img = images[camera_index][image_index]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (height, width), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(refined_corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (width, height), refined_corners, ret)
                cv2.imshow(f"camera_{camera_index} image_{image_index}", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints)
        error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix=cameraMatrix, distCoeffs=None)
        print(f"Root Mean Squared Error {error}")

        output.append({'error': error, 'K': mtx, 'distCoeffs': dist, 'rvecs': rvecs, 'tvecs':tvecs})

    return output


def make_transform(rvec, tvec, inv=False):
        # convert a rotation vector and translation vector to a 4x4 rigid body transformation.
        p = np.eye(4)
        rotmat = Rotation.from_rotvec(rvec[:, 0]).as_matrix()
        tvec = tvec[:, 0]
        p[:3, :3] = rotmat
        p[:3, 3] = tvec
        if inv:
            p = np.linalg.inv(p)
        return p

def make_o3d_pcd(bgr_image, depth_image, K, rvec, tvec):
    P = make_transform(rvec=rvec, tvec=tvec, inv=False)
    height, width = depth_image.shape
    camera_model = o3d.camera.PinholeCameraIntrinsic(
                                                    width=width,
													height=height,
                                                    fx=K[0,0],
                                                    fy=K[1,1],
                                                    cx=K[0,2],
													cy=K[1,2],
                                                )

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = o3d.geometry.Image(rgb_image)
    depth_image = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,depth_image,convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,camera_model, P)
    return pcd


def load_sequence(bagfile, process_msgs=True):
    bag = rosbag.Bag(bagfile)
    info = bag.get_type_and_topic_info()[1]
    data = []
    required_topics = info.keys()
    dct = {topic2key(topic): None for topic in required_topics}

    for topic, msg, ts in bag.read_messages():
        _, _, serial, modality, msg_type = topic.split('/')
        if process_msgs:
            if msg_type == 'image_raw':
                encoding = 'bgr8' if modality == 'color' else '16UC1'
                msg = bridge.imgmsg_to_cv2(msg, encoding)
            elif msg_type == 'camera_info':
                msg = {'K': np.array(msg.K).reshape((3, 3)),
                       'D': np.array(msg.D),
                       'R': np.array(msg.R).reshape((3,3)),
                       'P': np.array(msg.P).reshape((3,4)),
                       'height': msg.height,
                       'width': msg.width,
                       'distortion_model': msg.distortion_model,
                }
        dct[topic2key(topic)] = {'topic': topic, 'msg': msg, 'ts': ts}

        if not any([v is None for v in dct.values()]):
            data.append(dct)
            dct = {topic2key(topic): None for topic in required_topics}
    return data

def mean_reprojection_error(objpoints, imgpoints, rvec, tvec, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2,_ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    return mean_error/len(imgpoints)



class BagWrapper:
	def __init__(self, bagfile, num_calibration_images=5, limit=1000, serials =  ['101622073015', '138422073766']) -> None:
		self.data = load_sequence(bagfile=bagfile, process_msgs=True)
		self.data = self.data[:limit]
		dct = self.data[0]

		self.serials = serials
		# [*set(serial for serial,_,_ in dct.keys())]
		self.modalities = set(modality for _,modality,_ in dct.keys())
		self.suffixes = set(suffix for _,_,suffix in dct.keys())
		self.images = {serial:  {modality:[] for modality in self.modalities} for serial in self.serials}
		self.cams = {serial: {modality: {} for modality in self.modalities} for serial in self.serials}
		for frame, dct in enumerate(self.data):

			for (serial, modality, suffix),v in dct.items():
				if serial not in self.serials:
					continue
				if frame == 0 and suffix == "camera_info":
					print(serial, modality, suffix)
					intrinsics = {
						"width": int(v['msg']['width']),
						"height": int(v['msg']['height']),
						"fx": float(v['msg']['K'][0,0]),
						"fy": float(v['msg']['K'][1,1]),
						"cx": float(v['msg']['K'][0,2]),
						"cy": float(v['msg']['K'][1,2]),
					}
					self.cams[serial][modality]["Intrinsics"] = o3d.camera.PinholeCameraIntrinsic(**intrinsics)

				if suffix == 'image_raw':
					image = cv2.cvtColor(v['msg'], cv2.COLOR_BGR2RGB) if modality == 'color' else v['msg']
					self.images[serial][modality].append(image)

		for serial in self.serials:
			img = {i: self.images[serial]['color'][i] for i in range(num_calibration_images)}
			self.cams[serial]["Extrinsics"] = calibrate_extrinsics_pnp(img, num_calibration_images, self.cams[serial]['color']['Intrinsics'].intrinsic_matrix)

		self.colors = {serial:np.random.random(3) for serial in self.serials}

	def reprojection_error(self, points3d, points2d, cameraMatrix, rvec=None, tvec=None, distCoeffs=None):
		total_error = 0
		for i, (point2d, point3d) in enumerate(zip(points2d, points3d)):
			projected, _ = cv2.projectPoints(point3d, rvec, tvec, cameraMatrix, distCoeffs)
			reprojection_error = cv2.norm(point2d, projected, cv2.NORM_L2)/len(projected)
			total_error += reprojection_error

		mean_reprojection_error = total_error/len(point3d)

		return mean_reprojection_error

	def load_frame(self, n, color_coding=True):
		pcd_comb = o3d.geometry.PointCloud()

		for item, doc in self.cams.items():
			if item not in self.serials:
				print("invalid serial {serial}, skipping!!")
				continue
			# print(item, ":", doc)
			dep = o3d.geometry.Image(self.images[item]['aligned_depth_to_color'][n])
			col = o3d.geometry.Image(self.images[item]['color'][n])
			rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(col,dep,convert_rgb_to_intensity=False)
			pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,doc["color"]['Intrinsics'],doc['Extrinsics'])
			if color_coding:
				pc.paint_uniform_color(self.colors[item])
			pcd_comb += pc
		pcd_comb.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		# o3d.visualization.draw_geometries([pcd_comb])
		return pcd_comb




def main2(bagfile):
    data = load_sequence(bagfile, process_msgs=True)
    num_cameras = 4
    num_images = 3

    images = [[None for j in range(num_images)] for i in range(num_cameras)]
    depth = [[None for j in range(num_images)] for i in range(num_cameras)]
    K_depth = []
    K_color = []
    for i in range(num_images):
        d = data[i]
        j = 0
        for k, v in d.items():
            if k[1:] == ('aligned_depth_to_color', 'image_raw'):
                depth[j][i] = v['msg']

            if k[1:] == ('color', 'image_raw'):
                images[j][i] = v['msg']
                j += 1

        if len(K_depth) == 0:
            K_depth = [v['msg']['K'] for k, v in d.items() if k[1:] == ('aligned_depth_to_color', 'camera_info')]
        if len(K_color) == 0:
            K_color = [v['msg']['K'] for k, v in d.items() if k[1:] == ('color', 'camera_info')]

    output = calibrate_exrinsics(images, K_color, 0.04, 6, 9)

    pcds = []
    pcd_sum = o3d.geometry.PointCloud()
    for i in range(num_cameras):

        pcd = make_o3d_pcd(images[i][0], depth[i][0], K_depth[i], output[i]['rvecs'][0], output[i]['tvecs'][0])
        o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
        pcd_sum += pcd
    o3d.visualization.draw_geometries([pcd_sum])
    # import pdb; pdb.set_trace()


def main(bagfile, num_calibration_images=5, limit=1000, serials=['101622073015', '138422073766'], color_coding=False):
    bag_wrapper = BagWrapper(bagfile=bagfile , num_calibration_images=num_calibration_images, limit=limit, serials=serials)
    geometry = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
	# vis.add_geometry(geometry)
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    count = 0
    for i in range(1,len(bag_wrapper.data)):
        print(i)
        pcd = bag_wrapper.load_frame(i, color_coding=color_coding)
        geometry.points = pcd.points
        geometry.colors = pcd.colors
        # o3d.visualization.draw_geometries([geometry])
        # vis.remove_geometry(geometry)
        if count == 0:
            vis.add_geometry(geometry)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        count = count+1

    vis.close()



if __name__ == "__main__":
    bag = 'assets/bag_2022-05-11-14-37-56_0.bag'
    main(bag, color_coding=False)
