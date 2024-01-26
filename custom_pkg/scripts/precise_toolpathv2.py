import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import open3d as o3d
from scipy.spatial import KDTree
import transforms3d as t3d

decimal = 3
layers = 5

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
    min_x_point[2] = avg_z

    max_x_point = point_cloud[max_x_index].copy()
    max_x_point[2] = avg_z

    min_y_point = point_cloud[min_y_index].copy()
    min_y_point[2] = avg_z

    max_y_point = point_cloud[max_y_index].copy()
    max_y_point[2] = avg_z

    selected_points = [min_x_point, max_x_point, min_y_point, max_y_point]

    return selected_points

def set_rotation(point, target):
    # Calculate the direction vector from point to target
    direction = target - point
    
    # Normalize the direction vector
    direction /= np.linalg.norm(direction)
    
    # Calculate the rotation angles
    theta = np.arctan2(direction[1], direction[0])
    phi = np.arccos(direction[2])
    
    return theta, phi


# Change the file path to the location of your .ply file
ply_file_path = '/dev_ws/src/studio_II_project/custom_pkg/scripts/chair.ply'

cloud = o3d.io.read_point_cloud(ply_file_path)

# Convert Open3D PointCloud to sensor_msgs/PointCloud
all_points = np.asarray(cloud.points)
colors = np.asarray(cloud.colors) * 255.0  # Assuming colors are in the range [0, 255]
print(len(all_points))

# Assuming you have an array of points named 'all_points'
# Sort the points based on the z-value
all_points_sorted = all_points[np.argsort(all_points[:, 2])]

# Round the z-values to the 2nd decimal place
z_values = np.round(all_points_sorted[:, 2], decimals=decimal)

# Initialize a dictionary to store unique points for each group
grouped_points = {z: [] for z in np.unique(z_values)}

# Iterate through the points and assign them to the corresponding group
for point, z_value in zip(all_points_sorted, z_values):
    grouped_points[z_value].append(point)

# Sort points within each group based on x and y values
for z_value, group_points in grouped_points.items():
    grouped_points[z_value] = np.array(sorted(group_points, key=lambda x: (x[0], x[1])))

# Initialize a list to store the selected points for drawing lines
selected_points = []
# selected_points2 = []

# count = 0
# Iterate through the groups, print the data, and extract equidistant points
grouped_points_list = list(grouped_points.items())

interval = round(len(grouped_points) / layers)

for i in range(0, len(grouped_points), interval):
# for z_value, group_points in grouped_points.items():
    z, group_points = grouped_points_list[i]
    extremes = get_extreme_points(group_points)
    selected_points.append(extremes[0])
    selected_points.append(extremes[1])
    selected_points.append(extremes[2])
    selected_points.append(extremes[3])

# Convert the selected points to a NumPy array for 3D plotting
selected_points = np.array(selected_points)
# selected_points2 = np.array(selected_points2)

# Ensure selected_points is a 2D array
if len(selected_points.shape) == 1:
    selected_points = np.expand_dims(selected_points, axis=0)
# if len(selected_points2.shape) == 1:
#     selected_points2 = np.expand_dims(selected_points2, axis=0)
    
center = np.mean(all_points, axis=0)

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the selected points
ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], marker='o', color='red', label='Selected Points')
ax.scatter(center[0], center[1], center[2], marker='x', color='green', s=100, label='Center')
# ax.scatter(selected_points2[:, 0], selected_points2[:, 1], selected_points2[:, 2], marker='o', color='blue', label='Selected Points2')

# Separate points based on z values
z_values = set(point[2] for point in selected_points)


# Plot lines for points with the same z value
for z in z_values:
    z_points = [point for point in selected_points if point[2] == z]
    min_y_point = min(z_points, key=lambda p: p[1])
    max_y_point = max(z_points, key=lambda p: p[1])
    min_x_point = min(z_points, key=lambda p: p[0])
    max_x_point = max(z_points, key=lambda p: p[0])

    # Connect points with lines
    ax.plot([min_y_point[0], min_x_point[0], max_y_point[0], max_x_point[0], min_y_point[0]],
            [min_y_point[1], min_x_point[1], max_y_point[1], max_x_point[1], min_y_point[1]],
            [z, z, z, z, z], c='blue', marker='o')


for i in range(len(selected_points)):
    direction = center - selected_points[i]
    normalized_axis = np.cross([0, 0, 1], direction)  # Corrected axis calculation
    normalized_axis /= np.linalg.norm(normalized_axis)
    angle = np.arccos(np.dot([0, 0, 1], direction) / (np.linalg.norm([0, 0, 1]) * np.linalg.norm(direction)))
    rotation_matrix = t3d.axangles.axangle2mat(normalized_axis, angle)
    rotated_point = np.dot(rotation_matrix, selected_points[i])
    rotated_direction = np.dot(rotation_matrix, [0, 0, 1])
    ax.quiver(selected_points[i, 0], selected_points[i, 1], selected_points[i, 2], 
              rotated_direction[0], rotated_direction[1], rotated_direction[2],
              color='blue', length=0.1)
# # Set the rotation for each point
# rotations = [set_rotation(point, center) for point in selected_points]

# # Apply the rotation and plot the points after rotation
# for i, (theta, phi) in enumerate(rotations):
#     rotation_matrix = np.array([
#         [np.cos(theta), -np.sin(theta), 0],
#         [np.sin(theta), np.cos(theta), 0],
#         [0, 0, 1]
#     ]) @ np.array([
#         [np.cos(phi), 0, np.sin(phi)],
#         [0, 1, 0],
#         [np.sin(phi), 0, np.cos(phi)]
#     ])
#     rotated_point = rotation_matrix @ selected_points[i]
#     ax.quiver(selected_points[i, 0], selected_points[i, 1], selected_points[i, 2], 
#               rotated_point[0] - selected_points[i, 0], rotated_point[1] - selected_points[i, 1], rotated_point[2] - selected_points[i, 2],
#               color='blue', length=0.1, arrow_length_ratio=0.1)


# Setting labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('3D Plot with Selected Points')
plt.legend()
plt.show()