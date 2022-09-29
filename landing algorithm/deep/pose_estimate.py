import numpy as np
import sys,os
import math
import random
import time
from re import T
import cv2
from PIL import Image
from EDLinePython import EdgeDrawing as ED
from EDLinePython import LineDetector as LD
from deeplab import DeeplabV3
from ransac import fitLineRansac
from yolov5.yolo_module import YOLO
from feature_extraction import predict

fx = 1030
fy = 540
cx = 1030
cy = 540
camera_matrix = np.matrix([  [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0, 1]],  dtype=np.float32)

# UE4世界坐标系下的坐标
# 默认为左手系，坐标为NEU（北东上）（尝试）
gt_points = np.array([[-155.6, 0.8, 21.8911],
                    [98.3, 7.2, 21.8911],
                    [76.4, -15, 21.8911],
                    [-155.6, -20.4, 21.8911]],dtype=np.float32) 

gt_points1 = np.array([[0.8, -155.6, 21.8911],
                    [7.2, 98.3,  21.8911],
                    [- 15, 76.4, 21.8911],
                    [-20.4, -155.6, 21.8911]],dtype=np.float32)  

ned_origin_in_ue4_ref = np.array([[-134.4, -10.75, 22.8],
                                [-134.4, -10.75, 22.8],
                                [-134.4, -10.75, 22.8],
                                [-134.4, -10.75, 22.8]],dtype=np.float32)

gt_points = gt_points - ned_origin_in_ue4_ref
gt_points[:,2] = -gt_points[:,2]


def Quaternion2Euler(q):
    # yaw = heading
    w,x,y,z = q
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(2 * (w * y - z * x))
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [p,y,r]
def rotationMatrixToEulerAngles(rvecs):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvecs, R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    X = x*180.0/math.pi
    Y = y*180.0/math.pi
    Z = z*180.0/math.pi
    return X,Y,Z
# 经过三次旋转求解相机在世界坐标系中的坐标
def RotateByZ(Cx, Cy, Z):
    rz = Z*math.pi/180.0
    outX = math.cos(rz)*Cx - math.sin(rz)*Cy
    outY = math.sin(rz)*Cx + math.cos(rz)*Cy
    return outX, outY
def RotateByY(Cx, Cz, Y):
    ry = Y*math.pi/180.0
    outZ = math.cos(ry)*Cz - math.sin(ry)*Cx
    outX = math.sin(ry)*Cz + math.cos(ry)*Cx
    return outX, outZ
def RotateByX(Cy, Cz, X):
    rx = X*math.pi/180.0
    outY = math.cos(rx)*Cy - math.sin(rx)*Cz
    outZ = math.sin(rx)*Cy + math.cos(rx)*Cz
    return outY, outZ
    
def Euler2Quaternion(pose):
    # yaw = heading
    p, y, r = pose[:3]
    sinp = math.sin(p/2)
    siny = math.sin(y/2)
    sinr = math.sin(r/2)

    cosp = math.cos(p/2)
    cosy = math.cos(y/2)
    cosr = math.cos(r/2)
    w = cosr*cosp*cosy + sinr*sinp*siny
    x = sinr*cosp*cosy - cosr*sinp*siny
    y = cosr*sinp*cosy + sinr*cosp*siny
    z = cosr*cosp*siny - sinr*sinp*cosy
    return [w,x,y,z]
    
class PoseEstimate(object):
    def __init__(self):
        self.deeplab = DeeplabV3()
        self.yolo = YOLO()
    
    def estimate(self,image_path, is_show= False, save_path = None):
        # 读入2k尺寸的原图
        print("Loading image:",image_path)
        image = Image.open(image_path)
        # 航母目标检测
        yoloret = self.yolo.detect_image(image)

        if yoloret is None:
            print("yoloret is None")
            if save_path is not None:
                image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path,image)
            return None

        top_label,top_conf,top_boxes = yoloret
        max_score = -1
        # 得到得分最高的航母目标区域切片，以及切片在原图中的位置信息
        for i, c in list(enumerate(top_label)):
            predicted_class = self.yolo.class_names[int(c)]
            if predicted_class != "aircraft_carrier":
                print("wrong class:",predicted_class)
                continue
            box             = top_boxes[i]
            score           = top_conf[i]
            if score>max_score:
                top, left, bottom, right = box
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

        crop_image = image.crop([left, top, right, bottom])        
        segment_image = self.deeplab.detect_image(crop_image)
        rwpoints1,rwpoints2,boxA,boxB = predict(crop_image,segment_image)
        if rwpoints1 is None:
            print("rwpoints1 is none")
            if save_path is not None:
                image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path,image)
            return None

        trans_vec = np.array([left,top])

        for point in boxA[0]:
            point[0] = point[0]+trans_vec[0]
            point[1] = point[1]+trans_vec[1]
        for point in boxB[0]:
            point[0] = point[0]+trans_vec[0]
            point[1] = point[1]+trans_vec[1]
        for point in rwpoints1:
            point[0] = point[0]+trans_vec[0]
            point[1] = point[1]+trans_vec[1]
        for point in rwpoints2:
            point[0] = point[0]+trans_vec[0]
            point[1] = point[1]+trans_vec[1]
        # print("boxA[0]:",boxA[0])
        # print("boxB:",boxB)
        # print("rwpoints1:",rwpoints1)
        # print("rwpoints2:",rwpoints2)


        if is_show:
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            cv2.polylines(image,boxA,isClosed=True,color = [0,0,225])
            cv2.polylines(image,boxB,isClosed=True,color = [0,225,0])

            cv2.line(image, rwpoints1[0], rwpoints1[1],(0,255,255))
            cv2.line(image, rwpoints2[0], rwpoints2[1],(0,255,255))
            # cv2.imwrite("test111.png",image)
            cv2.imshow("image",image)
            if cv2.waitKey(0)==27:
                exit()
        if save_path is not None:
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            cv2.polylines(image,boxA,isClosed=True,color = [0,0,225],thickness=2)
            cv2.polylines(image,boxB,isClosed=True,color = [0,225,0],thickness=2)

            cv2.line(image, rwpoints1[0], rwpoints1[1],(0,255,255),3)
            cv2.line(image, rwpoints2[0], rwpoints2[1],(0,255,255),3)
            cv2.imwrite(save_path,image)

        estimate_points = []
        # 判断跑道线左右
        if rwpoints1[0][0]>rwpoints2[0][0]:
            rw_temp = rwpoints2.copy()
            rwpoints2 = rwpoints1
            rwpoints1 = rw_temp

        if rwpoints1[0][1]<rwpoints1[1][1]:
            estimate_points.append(rwpoints1[0])
            estimate_points.append(rwpoints1[1])
        else:
            estimate_points.append(rwpoints1[1])
            estimate_points.append(rwpoints1[0])
        if rwpoints2[0][1]<rwpoints2[1][1]:
            estimate_points.append(rwpoints2[1])
            estimate_points.append(rwpoints2[0])
        else:
            estimate_points.append(rwpoints2[0])
            estimate_points.append(rwpoints2[1])

        return np.array(estimate_points,dtype=np.float32)

    def calc_pnp(self,image_path):
        # 得到点的像素坐标
        image_points = self.estimate(image_path)
        # new_image_points = np.array([image_points[3],image_points[2],image_points[1],image_points[0]],dtype=np.float32)

        print("image_points:")
        print(image_points)
        # temp = gt_points[:,1].copy()
        # gt_points[:,1] = gt_points[:,0]
        # gt_points[:,0] = temp

        print("gt_points:")
        print(gt_points)        

        dist_coeffs = np.zeros((5, 1)) 
        (success, rotation_vector, translation_vector,inliners) = cv2.solvePnPRansac(gt_points, image_points, camera_matrix, dist_coeffs)
# ,flags=cv2.SOLVEPNP_P3P
        print("calc pnp:")
        print(success)
        print(rotation_vector)
        print(translation_vector)

        R = rotationMatrixToEulerAngles(rotation_vector)
        print("Estimate Euler:",R)
        x,y,z = translation_vector
        (x, y) = RotateByZ(x, y, -1.0*R[2])
        (x, z) = RotateByY(x, z, -1.0*R[1])
        (y, z) = RotateByX(y, z, -1.0*R[0])

        print("ned_position:")
        print("pos_x:",-1*x)
        print("pos_y:",-1*y)
        print("pos_z:",-1*z)
        quaternion = Euler2Quaternion(R)
        print("quaternion:",quaternion)


pose_estimate = PoseEstimate()
# image_path = "datasets/normal/images/img_SimpleFlight__0_1648988413026713000.png"
# 339.877	-21.2615	-80.4453
# pose_estimate.calc_pnp(image_path)
# pose_estimate.estimate(image_path,True)

start = time.time()
dir_path = "datasets/normal/images"
save_path = "datasets/landing/landing_result"
logfile ="datasets/normal/point_estimate_result.txt" 

filename = "datasets/normal/images/img_SimpleFlight__0_1648988513716490000.png"
pose_estimate.estimate(filename,False,save_path="datasets/feature_extraction_result2.png")

# with open(logfile,"w") as f:
#     f.flush()
#     f.write("FileName\tP0_X\tP0_Y\tP1_X\tP1_Y\tP2_X\tP2_Y\tP3_X\tP3_Y\n")

#     filelist = os.listdir(dir_path)
#     filelist.sort()
#     for filename in filelist:
#         estimate_points = pose_estimate.estimate(os.path.join(dir_path, filename))
#         if estimate_points is not None:
#             f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
#                 filename, estimate_points[0][0],estimate_points[0][1],                
#                 estimate_points[1][0],estimate_points[1][1],
#                 estimate_points[2][0],estimate_points[2][1],
#                 estimate_points[3][0],estimate_points[3][1]
#             ))
#         else:
#             f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
#                 filename, None, None, None, None, None, None, None, None
#             ))


# print("Total Time:",time.time()- start)
# print("Processed Images",len(os.listdir(dir_path)))


# gt_q = [-0.0336528,0.023956,-0.00202736,-0.999144]
# gt_euler = Quaternion2Euler(gt_q)
# print("gt Euler:", gt_euler)
# pose_estimate.estimate(image_path,True)