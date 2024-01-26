#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PlanningScene
from moveit_msgs.srv import GetPlanningScene
from moveit_commander import MoveGroupCommander
import numpy as np
import transforms3d as t3d
import open3d as o3d
from scipy.spatial import KDTree
from typing import List, Tuple
from moveit_msgs.msg import CollisionObject, PlanningScene, ObjectColor

from moveit_commander import PlanningSceneInterface

from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Point,
    Quaternion,
    Vector3,
)
from commander.msg import Goal
from commander.srv import (
    ExecuteTrajectory,
    PlanGoal,
    PlanGoalRequest,
    PlanSequence,
    PlanSequenceRequest,
    PickPlace,
    GetTcpPose,
    VisualizePoses,
    SetEe,
)

from commander.utils import poses_from_yaml, load_scene
from tf.transformations import euler_from_matrix, quaternion_from_euler
from shape_msgs.msg import Mesh
from geometry_msgs.msg import Point
from shape_msgs.msg import MeshTriangle
from std_msgs.msg import ColorRGBA


decimal = 3
layers = 5
offset = 0.15
move_x = 0.5  #just for testing
move_z = 0.1  #just for testing

rospy.init_node("reconstruction")

ply_file_path = rospy.get_param("~ply_file_path", "/dev_ws/src/software_II_project/custom_pkg/captures/raw.ply")

load_scene()

plan_goal_srv = rospy.ServiceProxy("commander/plan_goal", PlanGoal)
plan_sequence_srv = rospy.ServiceProxy("commander/plan_sequence", PlanSequence)
execute_trajectory_srv = rospy.ServiceProxy("commander/execute_trajectory", ExecuteTrajectory)
get_tcp_pose_srv = rospy.ServiceProxy("commander/get_tcp_pose", GetTcpPose)
set_ee_srv = rospy.ServiceProxy("commander/set_ee", SetEe)
pick_place_srv = rospy.ServiceProxy("commander/pick_place", PickPlace)

def create_mesh(cloud):
    # Downsample the point cloud
    downsampled_point_cloud = cloud.voxel_down_sample(voxel_size=0.01)

    # Perform Delaunay triangulation on the downsampled point cloud
    triangulation = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downsampled_point_cloud, alpha=0.1)

    # Assuming you have the correct triangulation.triangles data
    triangles_flat = [int(idx) for face in triangulation.triangles for idx in face]

    # Convert Open3D TriangleMesh to ROS Mesh message
    mesh_msg = Mesh()
    mesh_msg.vertices = [Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])) for pt in triangulation.vertices]

    # Create MeshTriangle messages for each triangle
    mesh_msg.triangles = [MeshTriangle(vertex_indices=[triangles_flat[i], triangles_flat[i + 1], triangles_flat[i + 2]]) for i in range(0, len(triangles_flat), 3)]

    return mesh_msg

def display_poses(poses: List[Pose], frame_id: str = "base_link") -> None:
    rospy.wait_for_service("/visualize_poses", timeout=10)
    visualize_poses = rospy.ServiceProxy("/visualize_poses", VisualizePoses)
    visualize_poses(frame_id, poses)

def process_target_poses(target_pose):
   
    # Plan the goal
    success = plan_goal_srv(Goal(pose=target_pose, vel_scale=0.2, acc_scale=0.2, planner='lin')).success

    # Check if planning is successful
    if success:
        # Execute the trajectory
        success = execute_trajectory_srv()

        # Check if execution is successful
        if not success:
            rospy.loginfo("Failed to execute trajectory")

    else:
        rospy.loginfo("Failed to plan")


def get_extreme_points(point_cloud):
    # Extract z coordinates from the point cloud
    z_coords = point_cloud[:, 2]
    
    # Calculate the average z value
    avg_z = np.mean(z_coords)

    # Find indices of points with smallest and largest x coordinates
    min_x_index, max_x_index = np.argmin(point_cloud[:, 0]), np.argmax(point_cloud[:, 0])

    # Find indices of points with smallest and largest y coordinates
    min_y_index, max_y_index = np.argmin(point_cloud[:, 1]), np.argmax(point_cloud[:, 1])

    # Extract and update the extreme points with the average z-coordinate
    min_x_point = point_cloud[min_x_index].copy()
    min_x_point[0] -= offset
    min_x_point[2] = avg_z

    max_x_point = point_cloud[max_x_index].copy()
    max_x_point[0] += offset
    max_x_point[2] = avg_z

    min_y_point = point_cloud[min_y_index].copy()
    min_y_point[1] -= offset
    min_y_point[2] = avg_z

    max_y_point = point_cloud[max_y_index].copy()
    max_y_point[1] += offset
    max_y_point[2] = avg_z

    selected_points = [min_x_point, max_x_point, min_y_point, max_y_point]

    return selected_points

def set_rotation(point, target):
    direction = target - point
    direction /= np.linalg.norm(direction)
    theta = np.arctan2(direction[1], direction[0])
    phi = np.arccos(direction[2])
    return theta, phi

def convert_to_pose(point):

    global center

    pose = Pose()

    # Set position
    pose.position.x = point[0] + move_x #remove constant when using actual scan
    pose.position.y = point[1] 
    pose.position.z = point[2] + move_z

    direction = center - point
    normalized_axis = np.cross([0, 0, 1], direction)  # Corrected axis calculation
    normalized_axis /= np.linalg.norm(normalized_axis)
    angle = np.arccos(np.dot([0, 0, 1], direction) / (np.linalg.norm([0, 0, 1]) * np.linalg.norm(direction)))
    rotation_matrix = t3d.axangles.axangle2mat(normalized_axis, angle)

    # Convert rotation matrix to Euler angles
    euler_angles = euler_from_matrix(rotation_matrix)

    # Convert Euler angles to quaternion
    quaternion = quaternion_from_euler(euler_angles[0], euler_angles[1], euler_angles[2])

    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose


# Load point cloud from a ply file
ply_file_path = '/dev_ws/src/custom_pkg/scripts/chair.ply'
cloud = o3d.io.read_point_cloud(ply_file_path)
all_points = np.asarray(cloud.points)

all_points_sorted = all_points[np.argsort(all_points[:, 2])]
z_values = np.round(all_points_sorted[:, 2], decimals=decimal)
grouped_points = {z: [] for z in np.unique(z_values)}

for point, z_value in zip(all_points_sorted, z_values):
    grouped_points[z_value].append(point)

for z_value, group_points in grouped_points.items():
    grouped_points[z_value] = np.array(sorted(group_points, key=lambda x: (x[0], x[1])))

selected_points = []

interval = round(len(grouped_points) / layers)

for i in range(0, len(grouped_points), interval):
    z, group_points = list(grouped_points.items())[i]
    extremes = get_extreme_points(group_points)
    selected_points.append(extremes[0])
    selected_points.append(extremes[1])
    selected_points.append(extremes[2])
    selected_points.append(extremes[3])

selected_points = np.array(selected_points)
if len(selected_points.shape) == 1:
    selected_points = np.expand_dims(selected_points, axis=0)

selected_points = selected_points[4:] # delete the first 4 points = floor points (remove this if object on top of a "table")

global center
center = np.mean(all_points, axis=0)

mesh = create_mesh(cloud)

scene_interface = PlanningSceneInterface()
scene = PlanningScene()
scene.is_diff = True

# ground cube
co = CollisionObject()
co.header.frame_id = 'base_link'
co.header.stamp = rospy.Time.now()
co.mesh_poses = [Pose(
    position=Point(move_x, 0.0, move_z),
    orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
)]
co.id = 'chair'
co.operation = CollisionObject.ADD
co.meshes.append(mesh)
scene.world.collision_objects.append(co)

oc = ObjectColor()
oc.id = 'chair'
oc.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.7)
scene.object_colors.append(oc)

scene_interface.apply_planning_scene(scene)

rospy.sleep(1)  # Wait for the scene to be updated

print("hello")

poses = []
success = set_ee_srv('rgb_camera_tcp')

# Assuming you have a loop to plan and execute for each pose
for point in selected_points:
    # Convert the selected point to a pose
    pose = convert_to_pose(point)
    poses.append(pose)

    process_target_poses(pose)

display_poses(poses)


