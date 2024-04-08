import numpy as np
# import habitat_sim

### Helper functions for handling coordinate frames ###
# There are 3 frames in use:
# 1) visualisation frame -- units are pixels, and this is
# the coordinate frame imshow uses to display the top-down map
# 2) map frame -- units are m, otherwise the frame is the
# the same as the visualisation frame
# 3) mesh frame -- this is the frame of the underlying scene mesh
# and hence also the frame used by the simulator, navigation etc.

def mapPoint2Pixel(point, bounds, res):
    px = (point[0] - bounds[0][0]) / res
    py = (point[2] + bounds[1][2]) / res
    return np.array([px, py])

def map2MeshFrame(point):
    return [point[0], point[1], -point[2]]

def mesh2MapFrame(point):
    return [point[0], point[1], -point[2]]

# def mesh2MapFrameQuaternion(q):
#     x, y, z, w = habitat_sim.utils.common.quat_to_coeffs(q)
#     flipped_q = habitat_sim.utils.common.quat_from_coeffs(
#         np.array([x, y, -z, -w])
#     )
#     return flipped_q

def convertMapFrame(map):
    return np.flipud(map) 

def calculateResolution(pathfinder, width):
    lower, upper = pathfinder.get_bounds()
    return abs(upper[2] - lower[2]) / float(width)