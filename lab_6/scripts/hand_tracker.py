import open3d as o3d
import cv2
import numpy as np
import mediapipe as mp
import torch
from manopth.manolayer import ManoLayer

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# per hand mano model tracker
class HandTrackerTorch:
    def __init__(self, side='right', device='cpu'):
        self.device = device
        self.side = side.lower()
        self.model = ManoLayer(mano_root='manopth/mano/models', use_pca=True, side=self.side).to(device)
        self.wrist_trans = None
        self.pose = None
        self.shape_params = None
        self.scale = None
        self.initialize_parameters()
        self.optimizer = torch.optim.Adam(params=[self.pose, self.wrist_trans], lr=0.1)
        self.scale_optimizer = torch.optim.Adam(params=[self.scale, self.wrist_trans, self.pose], lr=0.01)
        self.hand_mesh = self.make_hand_mesh()
        self._exist = False
    # check if hand is initialized
    @property
    def hand_exist(self):
        return self._exist
    # feed the current pose, and translation parameters to get the vertices and joints in 3D, then apply the scale
    def current_hand_output(self):
        verts, joints = self.model(th_pose_coeffs=self.pose, th_trans=self.wrist_trans)
        verts *= -1
        joints *= -1
        verts = verts/1000
        joints = joints/1000
        verts = verts * torch.abs(self.scale)
        joints = joints * torch.abs(self.scale)
        return verts, joints

    def current_hand_output_numpy(self):
        verts, joints = self.current_hand_output()
        return verts.detach().numpy(), joints.detach().numpy()
    # create mesh from the mano vertices and faces (triangles)
    def make_hand_mesh(self):
        verts, _ = self.current_hand_output_numpy()

        vertices = o3d.pybind.utility.Vector3dVector(verts[0])
        faces = o3d.pybind.utility.Vector3iVector(np.array(self.model.th_faces))
        mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=faces)
        mesh.paint_uniform_color([0, 0, 1] if self.side == 'right' else [1,0,0])
        return mesh
    # apply the new vertices to the mano mesh
    def update_hand_mesh(self):
        mano_verts, mano_joints = self.current_hand_output_numpy()

        self.hand_mesh.vertices = o3d.pybind.utility.Vector3dVector(mano_verts[0])
        self.hand_mesh.vertices

    def fit(self, joints3d, fit_scale=False, num_iterations=5, logging_iterations=1, logging_callback=None):
        # fit mano model to 3d joints
        self._exist = True
        joints3d = torch.Tensor(joints3d)
        losses = []
        for iteration in range(num_iterations):
            # disgard previous gradient calculations
            self.optimizer.zero_grad()
            self.scale_optimizer.zero_grad()

            # calculate the current hand joints, and vertices
            verts, jtr = self.current_hand_output()

            # apply MSE loss with the target from mediapipe hands
            loss = torch.sum((jtr[0] - joints3d) ** 2)
            #back-propagation
            loss.backward()
            # if 's' is pressed, fit scale
            if fit_scale:
                self.scale_optimizer.step()
            else:
                self.optimizer.step()
            
            losses.append(loss.item())
            # call the logging function
            if iteration % logging_iterations == 0:
                if logging_callback is not None:
                    logging_callback(loss.item())

        # plt.plot(losses)
        # plt.show()
        return verts, jtr

    # reset parameters
    def initialize_parameters(self):
        # note that we should not set parameters to zeros in order to break symmetry.
        self.wrist_trans = torch.rand([1, 3], requires_grad=True, device=self.device)
        self.pose = torch.rand([1, 48], requires_grad=True, device=self.device)
        self.shape_params = torch.zeros([1, 10], requires_grad=False, device=self.device)
        self.scale = torch.ones([1], requires_grad=True, device=self.device)
        # print(self.wrist_trans)
        # print(self.pose)

class HandsTracker:
    def __init__(self):
        self.sides = ['Left', 'Right']
        self.trackers = {side: HandTrackerTorch(side=side.lower()) for side in self.sides}

        self.mediapipe_detections = {side: None for side in self.sides}
        self.mediapipe_meshes = {side: [] for side in self.sides}
        self.mediapipe_results = {side: {'landmarks': None, 'handedness': None} for side in self.sides}
        self.mediapipe_overlay = None
        cv2.namedWindow("Mediapipe Overlay")

        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window("Hands Tracker")

    def _add_mano_mesh(self):
        # add mano meshes to the visualizer geometries.
        for side, tracker in self.trackers.items():
            self.visualizer.add_geometry(tracker.hand_mesh)

    def _update_mano_mesh(self):
        # update the mano mesh vertices
        for side, tracker in self.trackers.items():
            tracker.update_hand_mesh()

            self.visualizer.update_geometry(tracker.hand_mesh)

    def _add_mediapipe_joint_meshes(self):
        # create sphere meshes for the mediapipe joints
        for side, joint3d_mediapipe in self.mediapipe_detections.items():
            if joint3d_mediapipe is not None:
                if self.mediapipe_meshes[side] is not None:
                    for joint_mesh in self.mediapipe_meshes[side]:
                        self.visualizer.remove_geometry(joint_mesh)
                self.mediapipe_meshes[side] = []
                for joint3d in joint3d_mediapipe:
                    joint_mesh = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(joint3d, relative=False)
                    joint_mesh.paint_uniform_color([1, 0.5, 0] if side == 'Left' else [0, 0.5, 1])

                    self.mediapipe_meshes[side].append(joint_mesh)
                    self.visualizer.add_geometry(joint_mesh)
    # 
    def _update_mediapipe_joint_meshes(self):
        # translate the joint spheres to the new mediapipe joints
        for side, joint3d_mediapipe in self.mediapipe_detections.items():
            if joint3d_mediapipe is not None:
                for i, (joint3d, joint_mesh) in enumerate(zip(joint3d_mediapipe, self.mediapipe_meshes[side])):
                    joint_mesh.translate(joint3d, relative=False)
                    self.visualizer.update_geometry(joint_mesh)

    def update_visualizer(self):
        # update/initialize mediapipe joints and mano meshes
        for side, joint_meshes in self.mediapipe_meshes.items():
            if len(joint_meshes) > 0:
                self._update_mediapipe_joint_meshes()
                self._update_mano_mesh()
            elif self.mediapipe_detections[side] is not None:
                self._add_mediapipe_joint_meshes()
                self._add_mano_mesh()

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        if self.mediapipe_overlay is not None:
            # print(self.mediapipe_overlay)
            for side, mediapipe_result in self.mediapipe_results.items():
                mp_drawing.draw_landmarks(self.mediapipe_overlay, mediapipe_result['landmarks'],
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(
                                              color=[255, 0, 0] if side == 'Right' else [0, 0, 255]))
            cv2.imshow("Mediapipe Overlay", self.mediapipe_overlay)

    def process_mediapipe(self, bgr_image):
        joints3d = {}
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
            results = hands.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            overlay = bgr_image
            if results.multi_hand_landmarks is None or results.multi_handedness is None:
                return None, None

            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                classification = handedness.classification[0]
                score, label = classification.score, classification.label
                joints3d[label] = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                joints3d[label][:, [0,1]] *= -1
                self.mediapipe_detections[label] = joints3d[label]
                self.mediapipe_results[label] = {'landmarks': landmarks, 'handdedness': handedness}

        self.mediapipe_overlay = bgr_image
        return joints3d, results

    def process_mano(self, joints3d, init=False):
        for side, j3d in joints3d.items():
            self.trackers[side].process(j3d, init)

    def logging_callback(self, loss):
        print(loss)
        self.update_visualizer()

    def process(self, bgr_image, fit_scale):
        # extract mediapipe joints
        joints3d_mp, results = self.process_mediapipe(bgr_image)
        if joints3d_mp is not None:
            for side, joints3d in joints3d_mp.items():
                # set number of iterations to 1000 for the first time only, otherwise update only for 10 iterations
                num_iterations = 10 if self.trackers[side].hand_exist else 1000
                # how frequently to update logs
                logging_iterations = num_iterations//2
                self.trackers[side].fit(joints3d, fit_scale=fit_scale, num_iterations=num_iterations, logging_iterations=logging_iterations,
                                        logging_callback=self.logging_callback)
        # update visualizer with the new mediapipe, or mano meshes
        self.update_visualizer()

    def close(self):
        self.visualizer.destroy_window()



if __name__ == '__main__':
    # Initialize left, and right HandsTracker
    hands_tracker = HandsTracker()
    # open webcam video capture stream
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            # read next keyboard key (While opencv window is selected)
            k = cv2.waitKey(1)
            # quit if k is 'q'
            if k & 0xFF == ord('q'):
                break
            # process the frame. if k is 's' fit scale along the the pose, and translation otherwise, fit only pose and translation
            hands_tracker.process(frame, fit_scale=k == ord('s'))

    cap.release()
    cv2.destroyAllWindows()
    hands_tracker.close()
