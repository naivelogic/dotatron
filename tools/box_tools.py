
def poly2xywha(cx, cy, width, height, theta):
    """
    Check the angle in the OPENCV format for problems and record and change them
    """
    if theta == 0:
        theta = -90
        tmp = width
        width = height
        height = tmp
    
    if width != max(width, height):
        # width is not the longest edge
        theta = theta - 90
        return cx, cy, height, width, theta
    else:
        # width is the longest edge 
        return cx, cy, width, height, theta




