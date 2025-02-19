import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the numpy array (adjust the path and file format as needed)
gt_array = np.load('/data/ISAP_data_fig_table/ours_pred_09_sfm_depth_corr.npy')
sfmlearner_array = np.load('/data/ISAP_data_fig_table/ours_gt_09_sfm_depth_corr.npy')

# Extract x and y coordinates
gt_x = gt_array[:, 0]
gt_y = gt_array[:, 2]
gt_z = -gt_array[:, 1]
sfmlearner_x = sfmlearner_array[:, 0]
sfmlearner_y = sfmlearner_array[:, 2]
sfmlearner_z = -sfmlearner_array[:, 1]

fig = plt.figure(figsize=(5, 5))  # Width=4 inches, Height=3 inches



plt.scatter(gt_x, gt_y, s=1, c='red', label='GT')
plt.scatter(sfmlearner_x, sfmlearner_y, s=1, c='blue', label = 'Ours')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
plt.title('Ours 2D trajectory on KITTI odometry seq 09')
plt.legend()

# ax = fig.add_subplot(111, projection='3d')
# plt.scatter(gt_x, gt_y, gt_z, s=1, c='red', label='GT')
# plt.scatter(sfmlearner_x, sfmlearner_y, sfmlearner_z, s=1, c='blue', label = 'sfmlearner')
# ax.scatter(gt_x, gt_y, gt_z, s=1, c='red', label='GT')
# ax.scatter(sfmlearner_x, sfmlearner_y, sfmlearner_z, s=1, c='blue', label = 'sfmlearner')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_ylabel('Z-axis')
# ax.set_title('3D trajectory on KITTI odometry seq 09')




plt.grid(True)

# Save the plot as a PNG file
plt.savefig('/data/ISAP_data_fig_table/2d_traj_ours_odom_09_sfm.png', dpi=300, bbox_inches='tight')

# Optionally, you can also show the plot
# plt.show()
