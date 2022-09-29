from cProfile import run
import os
import random
import math
from re import T
from matplotlib.pyplot import thetagrids
import cv2
import numpy as np
from PIL import Image
from EDLinePython import EdgeDrawing as ED
from EDLinePython import LineDetector as LD
from deeplab import DeeplabV3
from ransac import fitLineRansac


def hough_transform(image_path,save_path):
    img = cv2.imread(image_path)
    H,W,C = img.shape
    # 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    canny = cv2.Canny(blur, 100, 200)
    # print(canny.shape)
    # 提取hough片段
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 40, minLineLength=5,maxLineGap=3)
    # lines = cv2.HoughLines(canny, 1, np.pi/180, 40)

    print("Number of hough transform lines:",len(lines))
    # 提取为二维
    lines1 = lines[:,0,:]
    # 绘制hough变换直线
    new_img = np.zeros((H,W,1))
    for x1,y1,x2,y2 in lines1[:]: 
        cv2.line(new_img,(x1,y1),(x2,y2),(255,0,0),1)
    cv2.imwrite(save_path,new_img)
    return 


def edline(image_path, save_path):
    ed = ED.EdgeDrawing()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges, _ = ed.EdgeDrawing(gray,smoothed=True)
    lines = LD.EDLine(edges=edges, minLineLen=5)

    print("number of edges:",len(edges))
    print(edges)
    print("number of lines:", len(lines))
    print(lines)

    # 仅仅画出每条线段头尾相接的线段。
    # final_lines = []
    # for line in lines:
    #     final_lines.append([line[0][1],line[0][0],line[-1][1],line[-1][0]])

    # for x1,y1,x2,y2 in final_lines[:]: 
    #     cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)

    # for line in lines:
    #     for point in line:
    #         cv2.circle(image,(point[1],point[0]),1,(255,0,0))
    # cv2.imwrite(save_path, image)
    return


def calc_dist(point, line_point1, line_point2):
    p1 = np.array(line_point1)
    p2 = np.array(line_point2)
    p = np.array(point)
    if np.linalg.norm(p2-p1)<0.01:
        return np.linalg.norm(p1-p)
    distance = np.abs(np.cross((p2-p1),(p-p1))) / np.linalg.norm(p2-p1)
    # print(distance)
    return distance


def process_lines(lines, threshold = 3, is_vertical = False):
    # 将hough变换得到的线段划分为左右两部分直线
    if not is_vertical:
        vertical_lines = []
        for line in lines:
            x1,y1,x2,y2 = line
            if 0.785 <= np.abs(math.atan2(y2-y1,x2-x1)) <= 2.356:
                vertical_lines.append(line)
    else:
        vertical_lines = lines
    a = []
    b = []
    c = []
    num = len(vertical_lines)
    if num < 2:
        print("vernum<2")
        return None

    a.append(vertical_lines[0])
    for i in range(1,num):
        print("process_lines")
        print("verline point:", vertical_lines[i][:2])

        dist1 = calc_dist(vertical_lines[i][:2], a[0][:2], a[0][-2:])
        dist2 = calc_dist(vertical_lines[i][-2:],a[0][:2], a[0][-2:])
        dist = min(dist1,dist2)
        if dist < threshold:
            a.append(vertical_lines[i])
        else:
            if len(b)==0:
                b.append(vertical_lines[i])
                continue
            dist1 = calc_dist(vertical_lines[i][:2], b[0][:2], b[0][-2:])
            dist2 = calc_dist(vertical_lines[i][-2:],b[0][:2], b[0][-2:])
            dist = min(dist1,dist2)
            if dist<threshold:
                b.append(vertical_lines[i])
            else:
                c.append(vertical_lines[i])
    if len(a)>= len(c) and len(b)>=len(c):
        return [a,b]
    elif len(b)>= len(a) and len(c)>=len(a):
        return [b,c]
    else:
        return [a,c]

def get_region(line_segments):
    sa = np.array(line_segments[0])
    sb = np.array(line_segments[1])
    aleft,aright = [sa[0][:2],sa[0][-2:]]
    bleft,bright = [sb[0][:2],sb[0][-2:]]
    for line in sa:
        point1, point2 = line[:2],line[-2:]
        if point1[0]<aleft[0]:
            aleft = point1
        if point1[0]>aright[0]:
            aright = point1
        if point2[0]<aleft[0]:
            aleft = point2
        if point2[0]>aright[0]:
            aright = point2
    for line in sb:
        point1, point2 = line[:2],line[-2:]
        if point1[0]<bleft[0]:
            bleft = point1
        if point1[0]>bright[0]:
            bright = point1
        if point2[0]<bleft[0]:
            bleft = point2
        if point2[0]>bright[0]:
            bright = point2
    aheight = math.ceil(math.sqrt((aleft[0]-aright[0])**2 +(aleft[1]-aright[1])**2)) + 10
    bheight = math.ceil(math.sqrt((bleft[0]-bright[0])**2 +(bleft[1]-bright[1])**2)) + 10
    # awidth = int(math.log(aheight,2))+10
    # bwidth = int(math.log(bheight,2))+10
    awidth = int(math.sqrt(aheight))+10
    bwidth = int(math.sqrt(bheight))+10
    ax = (aleft[0]+aright[0])/2
    ay = (aleft[1]+aright[1])/2
    bx = (bleft[0]+bright[0])/2
    by = (bleft[1]+bright[1])/2
    atheta = math.atan2(aleft[1]-aright[1], aleft[0]-aright[0]) * 180 / math.pi
    btheta = math.atan2(bleft[1]-bright[1], bleft[0]-bright[0]) * 180 / math.pi
    return [((ax,ay),(aheight,awidth),atheta), ((bx,by),(bheight,bwidth),btheta)]

def get_points_in_region(points, region):
    box = cv2.boxPoints(region)
    inner_points = []
    for point in points:
        pa = [box[0][0]-point[0],box[0][1]-point[1]]
        pb = [box[1][0]-point[0],box[1][1]-point[1]]
        pc = [box[2][0]-point[0],box[2][1]-point[1]]
        pd = [box[3][0]-point[0],box[3][1]-point[1]]
        ab = [box[1][0] - box[0][0], box[1][1] - box[0][1]]
        bc = [box[2][0] - box[1][0], box[2][1] - box[1][1]]
        cd = [box[3][0] - box[2][0], box[3][1] - box[2][1]]
        da = [box[0][0] - box[3][0], box[0][1] - box[3][1]]
        if np.min(np.sign(np.cross(pa,ab)), np.sign(np.cross(pb,bc)), np.sign(np.cross(pc,cd)), np.sign(np.cross(pd,da)))>=0:
            inner_points.append(point)
        elif np.max(np.sign(np.cross(pa,ab)), np.sign(np.cross(pb,bc)), np.sign(np.cross(pc,cd)), np.sign(np.cross(pd,da)))<=0:
            inner_points.append(point)           
    return inner_points

def is_point_in_region(point, region):
    box = cv2.boxPoints(region)
    pa = [box[0][0]-point[0],box[0][1]-point[1]]
    pb = [box[1][0]-point[0],box[1][1]-point[1]]
    pc = [box[2][0]-point[0],box[2][1]-point[1]]
    pd = [box[3][0]-point[0],box[3][1]-point[1]]
    ab = [box[1][0] - box[0][0], box[1][1] - box[0][1]]
    bc = [box[2][0] - box[1][0], box[2][1] - box[1][1]]
    cd = [box[3][0] - box[2][0], box[3][1] - box[2][1]]
    da = [box[0][0] - box[3][0], box[0][1] - box[3][1]]
    if min(np.sign(np.cross(pa,ab)), np.sign(np.cross(pb,bc)), np.sign(np.cross(pc,cd)), np.sign(np.cross(pd,da)))>=0:
        return True
    elif max(np.sign(np.cross(pa,ab)), np.sign(np.cross(pb,bc)), np.sign(np.cross(pc,cd)), np.sign(np.cross(pd,da)))<=0:
        return True
    return False

def box_to_contour(box):
    contour = []
    contour.append([[ box[0][0], box[0][1] ]])
    contour.append([[ box[1][0], box[1][1] ]])
    contour.append([[ box[2][0], box[2][1] ]])
    contour.append([[ box[3][0], box[3][1] ]])
    print(contour)
    return np.array(contour,dtype=np.float32)

def get_runway_point(runway, points, inline_threshold = 3):
    ref_point = np.array([runway[3],runway[2]])
    unit_vec = np.array([runway[1],runway[0]])
    min_val = 0
    max_val = 0
    for point in points:
        if np.abs(np.cross(unit_vec, np.array([point[1],point[0]])-ref_point)) > inline_threshold:
            continue
        val = np.dot(np.array([point[1],point[0]])-ref_point, unit_vec)
        if val<min_val:
            min_val = val
        if val>max_val:
            max_val = val
    # print("max_val:",max_val)
    # print("min_val:",min_val)
    max_point = max_val * unit_vec + ref_point
    min_point = min_val * unit_vec + ref_point
    return  np.array([max_point,min_point],dtype=int)


# def predict(image_path,save_path):
def predict(image,segment_image):
    # 加载模型，读入图像进行跑道线区域的预测
    # image = Image.open(image)
    # deeplab = DeeplabV3()
    # segment_image = deeplab.detect_image(image)
    # 转为cv2格式
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    segment_image = cv2.cvtColor(np.asarray(segment_image),cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # 提取分割区域边缘
    canny_image = cv2.Canny(segment_image,50,150)
    # 初步拟合分割区域边缘的跑道线
    lines = cv2.HoughLinesP(canny_image, 3, 3 * np.pi/180, 40, minLineLength=10,maxLineGap=40)
    if lines is None:
        print("lines is none")
        return [None,None,None,None]
    # print("Number of hough transform lines:",len(lines))
    # 提取为二维
    lines1 = lines[:,0,:]
    # for line in lines1:
    #     cv2.line(image,(line[1],line[0]), (line[1],line[0]),(0,0,255))

    # 得到左右两个备选区域
    linelist = process_lines(lines1, threshold=3, is_vertical=False)

    if linelist is None:
        print("linelist is none")
        return [None,None,None,None]
    if len(linelist[0]) == 0 or len(linelist[1]) == 0:    
        return [None,None,None,None]
    regions = get_region(linelist)
    regionA,regionB = regions
    # print("regionA:",regionA)W
    # print("regionB:",regionB)
    boxA = [cv2.boxPoints(regions[0]).astype(int)]
    boxB = [cv2.boxPoints(regions[1]).astype(int)]
    contourA = box_to_contour(boxA[0])
    contourB = box_to_contour(boxB[0])

    # 获取edline检测锚点
    ed = ED.EdgeDrawing()
    edges, _ = ed.EdgeDrawing(gray_image,smoothed=False)
    # print("number of edges:",len(edges))
    lines = LD.EDLine(edges=edges, minLineLen=10)

    # 提取区域内特征点
    sa_points = []
    sb_points = []
    for line in lines:
        for point in line:
            # 画edlines提取出的每一个点
            # cv2.circle(image,(point[1],point[0]),1,(255,0,0))
            if cv2.pointPolygonTest(contourA, [point[1],point[0]], measureDist=False) >= 0:
                sa_points.append(point)
            elif cv2.pointPolygonTest(contourB, [point[1],point[0]], measureDist=False) >= 0:
                sb_points.append(point)
            else:
                continue
    # print("sa_points:",len(sa_points)) 
    # print("sb_points:",len(sb_points))  
    # 注：fitLineRansac 函数输出的参数为[dy,dx,y0,x0]
    runway1 = fitLineRansac(np.array(sa_points,dtype=np.float32),1000,5)
    runway2 = fitLineRansac(np.array(sb_points,dtype=np.float32),1000,5)
    # print("runway1: ", runway1)
    # print("runway2: ", runway2)
    inline_thresholdA = math.log(regionA[1][0],2) - 2
    inline_thresholdB = math.log(regionB[1][0],2) - 2
    rwpoints1 = get_runway_point(runway1, sa_points, inline_thresholdA)
    rwpoints2 = get_runway_point(runway2, sb_points, inline_thresholdB)
    # print(rwpoints1)
    # print(rwpoints2)
    return  [rwpoints1,rwpoints2,boxA,boxB]

    cv2.line(image, rwpoints1[0], rwpoints1[1],(0,255,255))
    cv2.line(image, rwpoints2[0], rwpoints2[1],(0,255,255))

    # # 画出两个矩形框内的特征点
    # for point in sa_points:
    #     cv2.circle(image,(point[1],point[0]),1,(255,255,0))
    # for point in sb_points:
    #     cv2.circle(image,(point[1],point[0]),1,(255,255,0))

    # print(boxA)
    # print(boxB)
    # 画出预选择区域
    cv2.polylines(image,boxA,isClosed=True,color = [0,0,225])
    cv2.polylines(image,boxB,isClosed=True,color = [0,225,0])

    # cv2.imwrite(save_path,image)
    cv2.imshow("image",image)
    if cv2.waitKey(0)==27:
        exit()


# camera_params = {
#                 "PublishToRos": 1,
#                 "ImageType": 0,
#                 "Width": 2060,
#                 "Height": 1080,
#                 "FOV_Degrees": 90}
# fx = 2060/2/math.tan(math.pi/4)
# fy = 1080/2/math.tan(math.pi/4)
# cx = 1030
# cy = 540
# camera_matrix = np.array([  [fx, 0, cx],
#                             [0, fy, cy],
#                             [0,  0, 1]])

# gt_points = np.array([  [-152.0802, 3.8424,     21.8911],
#                         [-152.0802,-22.1575,    21.8911],
#                         [96.2197,   8.2424,     21.8911],
#                         [96.2197,   -22.1575,   21.8911]])

# hough_transform(image_path,save_path)
# edline(image_path,save_path)

def test_deeplab(image_path,save_path=None):
        # 加载模型，读入图像进行跑道线区域的预测
    deeplab = DeeplabV3()
    image = Image.open(image_path)
    segment_image = deeplab.detect_image(image)
    # 转为cv2格式
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    segment_image = cv2.cvtColor(np.asarray(segment_image),cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path,segment_image)
    cv2.imshow("i",segment_image)
    if cv2.waitKey(0)==27:
        exit()

def haiya(image):
    image = cv2.imread(image)
    # gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # 提取分割区域边缘
    canny_image = cv2.Canny(image,50,150)
    # 初步拟合分割区域边缘的跑道线
    cv2.imwrite("datasets/segment_canny_sample.png",canny_image)
    lines = cv2.HoughLinesP(canny_image, 3, 3 * np.pi/180, 40, minLineLength=10,maxLineGap=40)

    new_img = np.zeros(image.shape)
    lines1 = lines[:,0,:]
    for line in lines1:
        cv2.line(new_img,(line[1],line[0]), (line[1],line[0]),(0,0,255))
    
    cv2.imwrite("datasets/segment_hough_sample.png",new_img)



image_name = "datasets/segment_feature_extraction_sample.png"
haiya(image_name)