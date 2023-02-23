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

TAG_SIZE = 0.15244
FAMILIES = "tag16h5"
RES = (1280,720)

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

    cap = cv.VideoCapture(cap_device + cv.CAP_DSHOW)
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    at_detector: dt_apriltags.Detector = dt_apriltags.Detector(families=families, nthreads=nthreads, quad_decimate=quad_decimate, quad_sigma=quad_sigma, refine_edges=refine_edges, decode_sharpening=decode_sharpening, debug=debug)
    camera_info = {}
    # Camera Info Setup
    camera_info["res"] = RES

    camera_info["K"] = np.array([[736.4523760329718, 0.0, 636.8724945558921], [0.0, 741.0803279826814, 325.90101950102724], [0.0, 0.0, 1.0]])
    camera_info["params"] = [736.4523760329718, 741.0803279826814, 636.8724945558921, 325.90101950102724]
    camera_info["D"] = np.array([[-0.043073729765747026], [0.1660118413558553], [-0.9009625093041034], [1.2755476034926585]])

    camera_info["fisheye"] = True
    camera_info["map_1"], camera_info["map_2"] = cv.fisheye.initUndistortRectifyMap(camera_info["K"], camera_info["D"],
                                                                                 np.eye(3), camera_info["K"],
                                                                                 camera_info["res"], cv.CV_16SC2)

    while True:
        if not network_tables.isConnected():
            network_tables.waitForConnect()
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
        pose = [0, 0, 0]
        if size > 0:
            pose = np.array([metersToInches(pose_x_sum/size),metersToInches(pose_y_sum/size),metersToInches(pose_z_sum/size)])

        if USING_NT:
            network_tables.getEntry("jetson", "apriltags_pose").setDoubleArray(pose)

    cap.release()

if __name__ == '__main__':
    if USING_NT:
        network_tables.init()
        network_tables.waitForConnect()
    main()