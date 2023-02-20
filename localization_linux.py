import copy
import time
import argparse
import numpy as np
import apriltag
import cv2 as cv

# from pupil_apriltags import Detector
from tag import Tag

TAG_SIZE = 0.15244
FAMILIES = "tag16h5"
RES = (640,480)

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

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    at_options = apriltag.DetectorOptions(families=families, nthreads=nthreads, quad_decimate=quad_decimate, quad_blur=quad_sigma, refine_edges=refine_edges, refine_decode=decode_sharpening, debug=debug)
    at_detector = apriltag.Detector(at_options)
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

    

    elapsed_time = 1

    while True:
        start_time = time.time()
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        fwidth = (RES[0] + 31) // 32 * 32
        fheight = (RES[1] + 15) // 16 * 16
        # Load the Y (luminance) data from the stream
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
            undistorted_image = draw_tags(undistorted_image, detections, elapsed_time, pose)

        elapsed_time = time.time() - start_time

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        cv.imshow('AprilTags', undistorted_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
    pose
):
    for tag in tags:
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    fps = round(1.0 / elapsed_time)
    cv.putText(image,
               "FPS:" + '{:.1f}'.format(fps),
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)
    cv.putText(image,
               ("Pose: " + str(round(metersToInches(pose[0]),3)) + " " + str(round(metersToInches(pose[1]),3)) + " " + str(round(metersToInches(pose[2]),3))),
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)
    

    return image


if __name__ == '__main__':
    main()