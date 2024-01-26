import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def read_ply_file(file_path):
    cloud = o3d.io.read_point_cloud(file_path)
    return cloud


def main():
    #rospy.init_node('ply_to_pointcloud_publisher', anonymous=True)
    
    # Change the file path to the location of your .ply file
    ply_file_path = '/dev_ws/src/custom_pkg/scripts/chair.ply'

    o3d_cloud = read_ply_file(ply_file_path)

    # Convert Open3D PointCloud to sensor_msgs/PointCloud
    all_points = np.asarray(o3d_cloud.points)
    colors = np.asarray(o3d_cloud.colors) * 255.0  # Assuming colors are in the range [0, 255]
    print(len(all_points))

    # Convert the interval to meters (1 meter = 100 centimeters)
    interval_meters = 500

    # Initialize an empty list to store selected points
    selected_points = []

    # # Iterate through the original points and select points at the specified interval
    # for x, y, z in points:
    #     Check if the current point satisfies the interval condition
    #     if round(x % interval_meters) == 0 and round(y % interval_meters) == 0 and round(z % interval_meters) == 0:
    #         selected_points.append((x, y, z))

    # interval = round(len(points) / 50)

    # for i in range(0, len(points), interval):
    #     selected_points.append(points[i])


    # print(len(selected_points))

    # Number of points you want to extract for each z-value
    num_points_per_z = 10

    # Sort the points based on the z-value
    all_points_sorted = all_points[np.argsort(all_points[:, 2])]

    # Round the z-values to the 2nd decimal place
    z_values = np.round(all_points_sorted[:, 2], decimals=2)

    # Initialize a dictionary to store unique points for each group
    grouped_points = {z: [] for z in np.unique(z_values)}

    # Iterate through the points and assign them to the corresponding group
    for point, z_value in zip(all_points_sorted, z_values):
        grouped_points[z_value].append(point)

    # Sort points within each group based on x and y values
    for z_value, group_points in grouped_points.items():
        grouped_points[z_value] = np.array(sorted(group_points, key=lambda x: (x[0], x[1])))

    # Initialize a list to store the equidistant points
    equidistant_points = []

    # Iterate through the groups, print the data, and extract equidistant points
    for z_value, group_points in grouped_points.items():
        # print(f"Group with z-value {z_value}:")
        
        # # Print each unique point in the group
        # for point in group_points:
        #     print(point)

        # # Add a separator between groups for better readability
        # print("-" * 20)

        # Extract equidistant points for each z-value
        indices = np.where(z_values == z_value)[0]
        points = all_points_sorted[indices]

        if points.shape[0] == 1:
            equidistant_points.append(points[0])
        else:
            num_points_per_z = 5  # You can adjust this value as needed
            interval = max(1, points.shape[0] // num_points_per_z)  # Ensure interval is at least 1
            equidistant_points.extend(points[::interval][:num_points_per_z])


    # Optional: Create a new point cloud with the filtered and translated points
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(equidistant_points)

    # Save or visualize the result as needed
    o3d.visualization.draw_geometries([new_point_cloud])

if __name__ == '__main__':
    main()

