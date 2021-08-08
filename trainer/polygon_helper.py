import math
import numpy as np

def poly2xywha(bbox):
    """
    bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: BoxMode.XYWHA_ABS format [cx, cy, w, h, theta]
    """
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.reshape(bbox, newshape=(2, 4), order="F")

    bbox_shift = np.roll(bbox, 1, axis=1) 
    dx = - (bbox[0,:] - bbox_shift[0,:])
    dy = bbox[1,:] - bbox_shift[1,:]
    sides = np.hypot(dx, dy) # side lengths

    angles = np.arctan2(dy, dx)
    theta = np.rad2deg(angles[[0,2]].mean())

    c = np.rint(np.mean(bbox, axis=1))
    wh = np.rint((sides[0:2] + sides[2:4]) / 2.0)

    return [c[0],c[1], wh[1], wh[0], theta]
