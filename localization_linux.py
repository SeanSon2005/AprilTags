import copy
import argparse
import numpy as np
import dt_apriltags
import cv2 as cv
import time

# from pupil_apriltags import Detector
from tag import Tag

USING_NT = False

if USING_NT:
    import network_tables

# time.sleep(30)

TAG_SIZE = 0.15244
FAMILIES = "tag16h5"
RES = (640,480)

def get_cap(cap_device):
    try:
        return True, cv.VideoCapture(cap_device)
    except:
        return False, None
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="/dev/video0")
    parser.add_argument("--width", help='cap width', type=int, default=RES[0])
    parser.add_argument("--height", help='cap height', type=int, default=RES[1])

    parser.add_argument("--families", type=str, default=FAMILIES)
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args

def metersToInches(meters):
    return meters * 39.3701

def main():
    definedTags = Tag(TAG_SIZE, FAMILIES)

    # Add information about tag locations THIS ARE GLOBAL LOCATIONS IN INCHES
    # Function Arguments are id,x,y,z,theta_x,theta_y,theta_z
    definedTags.add_tag(1, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(2, 12., 0., 0., 0., 0., 0.)
    definedTags.add_tag(3, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(4, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(5, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(6, 0., 0., 0., 0., 0., 0.)


    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug
    #while True:
    #    print("hehehe")
    r, cap = get_cap(cap_device)
    # while not r:
        #time.sleep(1)
        #r, cap = get_cap(cap_device)
    #fi = open("wehavecap.txt", "a")
    #fi.write("we have cap")
    #fi.close()
    # while not cap.isOpened():
        #time.sleep(1)
        #r, cap = get_cap(cap_device)
    #fil = open("wehaveopenedcap.txt", "a")
    #fil.write("we have opened the cap")
    #fil.close()
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    at_detector: dt_apriltags.Detector = dt_apriltags.Detector(families=families, nthreads=nthreads, quad_decimate=quad_decimate, quad_sigma=quad_sigma, refine_edges=refine_edges, decode_sharpening=decode_sharpening, debug=debug)
    camera_info = {}
    # Camera Info Setup
    camera_info["res"] = RES

    camera_info["K"] = np.array([[367.7013393230449, 0.0, 323.3378629663504], [0.0, 369.7151984089531, 162.63699072828888], [0.0, 0.0, 1.0]])
    camera_info["params"] = [367.7013393230449, 369.7151984089531, 323.3378629663504, 162.63699072828888]
    camera_info["D"] = np.array([[-0.042203858496260044], [-0.08378810354583231], [0.4607572694660925], [-0.5671615907615343]])

    camera_info["fisheye"] = True
    camera_info["map_1"], camera_info["map_2"] = cv.fisheye.initUndistortRectifyMap(camera_info["K"], camera_info["D"],
                                                                                 np.eye(3), camera_info["K"],
                                                                                 camera_info["res"], cv.CV_16SC2)

    while True:
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)
        #undistort image
        undistorted_image = cv.remap(debug_image, camera_info["map_1"], camera_info["map_2"], interpolation=cv.INTER_LINEAR)
        gray = cv.cvtColor(undistorted_image,cv.COLOR_BGR2GRAY)

        tags = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_info["params"],
            tag_size=TAG_SIZE,
        )
        
        detections = []
        pose_x_sum = 0
        pose_y_sum = 0
        pose_z_sum = 0
        for detection in tags:
            if detection.tag_id < 1 or detection.tag_id > 9 or detection.decision_margin < 20:
                continue
            detections.append(detection)
            curPose = definedTags.estimate_pose(detection.tag_id, detection.pose_R, detection.pose_t)
            if curPose is not None:
                pose_x_sum += curPose[0][0]
                pose_y_sum += curPose[1][0]
                pose_z_sum += curPose[2][0]

        size = len(detections)
        if size > 0:
            pose = np.array([pose_x_sum/size,pose_y_sum/size,pose_z_sum/size])

        poseX = pose[0][0]
        poseY = pose[1][0]
        poseZ = pose[2][0]

        poseUpload = [poseX, poseY, poseZ]

        if USING_NT:
            network_tables.getEntry("jetson", "apriltags_pose").setDoubleArray(poseUpload)
        # cv.imshow('AprilTags', undistorted_image)

    cap.release()
    #cv.destroyAllWindows()

if __name__ == '__main__':
    if USING_NT:
        while not network_tables.isConnected():
            time.sleep(0.3)
    main()