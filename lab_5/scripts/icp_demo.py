import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform.rotation import Rotation

# read source point cloud
pcd_source = o3d.io.read_point_cloud('assets/models/oats/texturedMesh_alligned_vertex_color.ply')
pcd_source = pcd_source.voxel_down_sample(0.005)
# visualize the source point cloud
o3d.visualization.draw_geometries([pcd_source])
# create a mock target point cloud as a toy example
points = np.array(pcd_source.points)
extents = points.max(axis=0) - points.min(axis=0)
print(extents)
translation = 0.05 * extents
transform = np.eye(4)
transform[:3,3] = translation
pcd_target = o3d.geometry.PointCloud(pcd_source)
transform[:3,:3] = Rotation.from_euler('xyz', [25,0,0], degrees=True).as_matrix()
transform[:3,3] = [0,0,0.1]
pcd_target.transform(transform)

# set color for source to red and target to blue (later solution is green)
pcd_source.paint_uniform_color([1,0,0])
pcd_target.paint_uniform_color([0,1,0])
pcd_solution = o3d.geometry.PointCloud(pcd_source)
pcd_solution.paint_uniform_color([0,0,1])

o3d.visualization.draw_geometries([pcd_source, pcd_target])

# Note it is better to run iCP once with ICPConvergenceCriteria(max_iteration=num_iteration) but here we want to visualize each step.
num_iterations = 30
initial_transformation = np.eye(4)
threshold = 0.05 * max(extents)


vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_source)
vis.add_geometry(pcd_target)
vis.add_geometry(pcd_solution)

ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])
# geometries = [pcd_source, pcd_target, pcd_solution]
# o3d.visualization.draw_geometries(geometries)
cumulative_transformation = np.eye(4)
errors = []
errors2 = []
for iteration in range(num_iterations):
    reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd_solution, pcd_target, threshold, initial_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),)
    
    solution_transform = reg_p2p.transformation
    cumulative_transformation = cumulative_transformation @ solution_transform
    diff = (np.array(pcd_solution.points) - np.array(pcd_target.points))
    # exact_rmse = np.mean(np.sum(diff ** 2, axis=1)**0.5)
    exact_rmse = mean_squared_error(np.array(pcd_solution.points), np.array(pcd_target.points), squared=False)
    errors.append(exact_rmse) # calculation could be slightly different from rmse in open3d
    errors2.append(reg_p2p.inlier_rmse)
    print(f"Iteration registration errors: Inlier MSE is {reg_p2p.inlier_rmse}, Inlier Fitness is {reg_p2p.fitness}")
    print(f"RMSE(same object) = {exact_rmse}")
    print(f"cumulative translation = {cumulative_transformation[:3,3]} cumulative rotation = {Rotation.from_matrix(cumulative_transformation[:3,:3]).as_euler('xyz', degrees=True)}")
    initial_transformation = solution_transform
    pcd_solution = pcd_solution.transform(solution_transform)
    vis.update_geometry(pcd_solution)
    vis.poll_events()
    vis.update_renderer()
    # if iteration%5 == 0:
    #     o3d.visualization.draw_geometries(geometries)

vis.close()
plt.plot(np.arange(num_iterations), errors)
plt.plot(np.arange(num_iterations), errors2)

plt.xlabel('num iterations')
plt.ylabel('error(RMSE)')
plt.show()