import math
import numpy as np
#from DOTA_devkit import polyiou

def polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x0, y0, x1, y1, x2, y2, x3, y3]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(2,4),order='F')
    angle =  math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])

    center = [[0],[0]]

    for i in range(4):
        center[0] += bbox[0,i]
        center[1] += bbox[1,i]

    center = np.array(center,dtype=np.float32)/4.0

    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose(),bbox-center)

    xmin = np.min(normalized[0,:])
    xmax = np.max(normalized[0,:])
    ymin = np.min(normalized[1,:])
    ymax = np.max(normalized[1,:])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    return [float(center[0]),float(center[1]),w,h,angle]

def hrbb2hw_obb(proposal, hw):
    hrbb_x_min = proposal[0]
    hrbb_y_min = proposal[1]
    hrbb_x_max = proposal[2]
    hrbb_y_max = proposal[3]  
    
    W = hrbb_x_max - hrbb_x_min
    H = hrbb_y_max - hrbb_y_min

    h = hw[3]
    w = hw[2]  
    h2 = H - h
    w2 = W - w
    
    obb_pt_1 = np.array([hrbb_x_min + w, hrbb_y_min])
    obb_pt_2 = np.array([hrbb_x_max, hrbb_y_min + h2])
    obb_pt_3 = np.array([hrbb_x_min + w2, hrbb_y_max])
    obb_pt_4 = np.array([hrbb_x_min, hrbb_y_min + h])
    
    obb_bbox = np.array([
            obb_pt_1[0], obb_pt_1[1],
            obb_pt_2[0], obb_pt_2[1],
            obb_pt_3[0], obb_pt_3[1],
            obb_pt_4[0], obb_pt_4[1]
        ])
    
    return obb_bbox
