import cv2
import numpy as np
# from scipy.signal import wiener
import os

# def MSRCP(img, sigma_list=[15, 80, 270], low_clip=0.001, high_clip=0.999):
#     img = img.astype(np.float32)
#     img[img == 0] = 1e-6

#     intensity = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
#     retinex = np.zeros_like(intensity)
#     for sigma in sigma_list:
#         blur = cv2.GaussianBlur(intensity, (0, 0), sigma)
#         retinex += np.log(intensity + 1e-6) - np.log(blur + 1e-6)
#     retinex /= len(sigma_list)

#     # 拉伸动态范围
def simplest_color_balance(img, low_clip, high_clip,max_v):
    total = img.shape[0] * img.shape[1]
    flat = np.sort(img.flatten())
    low_val = flat[int(total * low_clip)]
    high_val = flat[int(total * high_clip)]
    img = np.clip((img - low_val) / (high_val - low_val), 0, 1)*max_v
    return img

#     retinex = simplest_color_balance(retinex, low_clip, high_clip)

#     # # 保留颜色信息
#     # img_sum = np.sum(img, axis=2, keepdims=True)
#     # img_sum[img_sum == 0] = 1e-6  # 避免除零
#     # img_color = img / img_sum
#     result = img * retinex[..., np.newaxis] 
#     result = np.clip(result, 0, 255).astype(np.uint8)

#     gamma = 0.9  # <1会增亮，>1会变暗，尝试不同值
#     result = np.power(result / 255.0, gamma) * 255.0
#     result = np.clip(result, 0, 255).astype(np.uint8)
#     # result=cv2.convertScaleAbs(img,alpha=1.1,beta=0)
#     return result


def single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log1p(img) - np.log1p(blur)

def multi_scale_retinex(intensity, sigmas):
    retinex = np.zeros_like(intensity, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(intensity, sigma)
    return retinex / len(sigmas)

def gamma_correction(img,gamma):
    table=np.array([((i/255.0)**gamma)*255 for i in range(256)],dtype=np.uint8)
    img=cv2.LUT(img,table)
    return img

def MSRCP(img, sigmas=[35, 50, 100]):
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # h,s,v=cv2.split(img)
    # print(h.shape)
    # print(s.shape)
    # print(v.shape)
    mean_v=np.mean(img)
    img=img.astype(np.float32)
    # img = img.astype(np.float32) + 1.0
    # intensity = np.mean(img, axis=2)
    max_img=np.max(img)
    r_img=img
    # r_img = multi_scale_retinex(img, sigmas)

    # intensity_retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    # r_img=simplest_color_balance(r_img,0.05,0.98,max_img).astype(np.uint8)
    # 色彩保留：用增强后的亮度替代原图强度
    # print(r_v.shape)
    r_img=r_img.astype(np.uint8)
    new_mean=np.mean(r_img)

    # img1=cv2.merge((h,s,r_v))

    gamma=np.log(190/255)/np.log(new_mean/255)
    r_img=gamma_correction(r_img,gamma)
    # img=cv2.merge((h,s,r_v))
    
    return r_img


def equal(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    equalized = cv2.equalizeHist(l)
    img=cv2.merge((equalized,a,b))
    img=cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
    return img


def CLAHE_and_wiener(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(3, 3))
    cl = clahe.apply(l)

    # wienered=wiener(cl,(5,5))
    # wienered = np.nan_to_num(wienered, nan=0.0, posinf=255, neginf=0)

    wienered=np.clip(cl,0,255)
    wienered=wienered.astype(np.uint8)

    limg=cv2.merge((wienered,a,b))

    mean=np.mean(wienered)
    gamma=np.log(170/255)/np.log(mean/255)
    wienered=gamma_correction(wienered,gamma)
    newlimg=cv2.merge((wienered,a,b))

    clahe_bgr=cv2.cvtColor(limg,cv2.COLOR_LAB2BGR)
    newclahe_bgr=cv2.cvtColor(newlimg,cv2.COLOR_LAB2BGR)
    return clahe_bgr,newclahe_bgr

def biFilter(img):
    return cv2.bilateralFilter(img,31,75,75)

def laplacian(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    lap=cv2.Laplacian(l,cv2.CV_64F)
    l_sharp=np.clip(l.astype(np.float64)-1.5*lap,0,255).astype(np.uint8)
    img=cv2.merge((l_sharp,a,b))
    img=cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
    return img

def unsharp_musk(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(img)
    blurred=cv2.GaussianBlur(l,(3,3),sigmaX=50)
    mask=l-blurred
    sharpened=np.clip(l+0.5*mask,0,255).astype(np.uint8)
    img=cv2.merge((sharpened,a,b))
    img=cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
    return img

def color_mask_redandblue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- 红色掩膜（两段） ---
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([40, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red1=cv2.bitwise_not(mask_red1)
    lower_red2 = np.array([140, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red2=cv2.bitwise_not(mask_red2)

    red_mask = cv2.bitwise_and(mask_red1, mask_red2)
    # red_mask=cv2.bitwise_not(red_mask)

    # --- 蓝色掩膜 ---
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask=cv2.bitwise_not(blue_mask)
    # --- 合并红色和蓝色掩膜 ---
    mask_combined = cv2.bitwise_and(red_mask, blue_mask)

    result = cv2.bitwise_and(img, img, mask=mask_combined)
    y_coords,x_coords=np.where(result==0)
    points=np.column_stack((x_coords,y_coords))
    corners=np.zeros((4,2),dtype=np.float32)
    sum_xy=points.sum(axis=1)
    diff_xy=points[:,0]-points[:,1]
    corners[0]=points(np.argmin(sum_xy))
    corners[1]=points(np.argmax(diff_xy))
    corners[2]=points(np.argmax(sum_xy))
    corners[3]=points(np.argmin(diff_xy))


    # img=cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return result

def color_mask_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义白色的范围（低饱和度、高亮度）
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 80, 255])

    # 创建白色掩膜
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 可视化白色掩膜（白色区域为255，其他为0）
    cv2.imshow('White Mask', mask_white)
    cv2.waitKey(0)

    # 可选：将白色区域提取出来（在原图中）
    white_only = cv2.bitwise_and(img, img, mask=mask_white)
    return white_only


def binarythresh(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def closing(img,kernel=15):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(img)
    k=cv2.getStructuringElement(cv2.MORPH_CROSS,(kernel,kernel))
    l= cv2.morphologyEx(l, cv2.MORPH_CLOSE, k)
    img= cv2.merge((l,a,b))
    return cv2.cvtColor(img,cv2.COLOR_LAB2BGR)

def opening(img,kernel=15):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(img)
    k=cv2.getStructuringElement(cv2.MORPH_CROSS,(kernel,kernel))
    l= cv2.morphologyEx(l, cv2.MORPH_OPEN, k)
    img= cv2.merge((l,a,b))
    return cv2.cvtColor(img,cv2.COLOR_LAB2BGR)

def Gaussian_blur(img):
    kernel=9
    sigma=15
    return cv2.GaussianBlur(img,(kernel,kernel),sigma)

def compute_quadrilateral_angles(pts):
    """
    pts: shape = (4, 2), 顺序为顺时针或逆时针排列的四边形四个点
    返回：每个角的内角（单位：度）
    """
    def angle_between_vectors(v1, v2):
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot)
        return np.degrees(angle)

    angles = []
    for i in range(4):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        angle = angle_between_vectors(v1, v2)
        angles.append(angle)
    
    return angles

def legal_region(area,approx,):
    if area >35000 and area <420*420:
        points=approx.reshape(-1,2)
        # sum_xy=points.sum(axis=1)
        # diff_xy=points[:,0]-points[:,1]
        # corners=np.zeros((4,2),dtype=np.float32)
        # corners[0]=points[np.argmin(sum_xy)]
        # corners[1]=points[np.argmax(diff_xy)]
        # corners[2]=points[np.argmax(sum_xy)]
        # corners[3]=points[np.argmin(diff_xy)]
        # if corners[0][0]<50 or corners[0][1]<50:
        #     return False
        # if corners[1][0]>650 or corners[1][1]<50:
        #     return False
        # if corners[2][0]>650 or corners[2][1]>650:
        #     return False
        # if corners[3][0]<50 or corners[3][1]>650:
        #     return False
        mean=np.sum(points,0)/4
        # flag=False
        if  160<mean[0]<480 and 160<mean[1]<480:
            # angles=compute_quadrilateral_angles(points)
            # for i in range(4):
            #     if abs(angles[i]-90)>25:
            #         flag=True
            # if flag:
            #     corners[0][0]=corners[3][0]=(corners[0][0]+corners[3][0])//2
            #     corners[0][1]=corners[1][1]=(corners[0][1]+corners[1][1])//2
            #     corners[3][1]=corners[2][1]=(corners[3][1]+corners[2][1])//2
            #     corners[1][0]=corners[2][0]=(corners[1][0]+corners[2][0])//2
            #     for i in range(4):
            #         points[i]=corners[i]
            #     approx=points.reshape(-1,1,2)
            print(0)
            return True
        print(1)
        return False
    else:
        print(2)
        return False

# def legal_region_multi(area,approx):
#     if area >20000 and area <450*450:
#         points=approx.reshape(-1,2)
#         # sum_xy=points.sum(axis=1)
#         # diff_xy=points[:,0]-points[:,1]
#         # corners=np.zeros((4,2),dtype=np.float32)
#         # corners[0]=points[np.argmin(sum_xy)]
#         # corners[1]=points[np.argmax(diff_xy)]
#         # corners[2]=points[np.argmax(sum_xy)]
#         # corners[3]=points[np.argmin(diff_xy)]
#         # if corners[0][0]<50 or corners[0][1]<50:
#         #     return False
#         # if corners[1][0]>650 or corners[1][1]<50:
#         #     return False
#         # if corners[2][0]>650 or corners[2][1]>650:
#         #     return False
#         # if corners[3][0]<50 or corners[3][1]>650:
#         #     return False
#         mean=np.sum(points,0)/points.shape[0]
#         if  200<mean[0]<500 and 200<mean[1]<500:
#             return True
#         return False
#     else:
#         return False

# if __name__=="__main__":
#     source=r"E:\2024-2025summer\CV\rotated_and_affine\corp3" 
#     parent_path=os.path.dirname(source)
#     path=os.path.join(parent_path,"crop3__")
#     os.makedirs(path,exist_ok=True)
#     for name in os.listdir(source):
#         # print(name)
#         img=cv2.imread(os.path.join(source,name))
#         img=closing(img)
#         img=opening(img)
        
#         img=MSRCP(img)
#         img=laplacian(img)
#         cv2.imwrite(fr"E:\2024-2025summer\CV\Retinexed_\{name}", img)
#         # img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#         # print(img_hsv[img.shape[0]//2][img.shape[1]//2])
#         # cv2.imshow('',img)
#         # cv2.waitKey(0)
#         '''yolo'''


#         # # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#         # # print(img[0][0])
#         # # print(img[170][145])ff
#         # img=binarythresh(img)
#         # # img=color_mask_white(img)
#         # # img=equal(img)

#         # img=biFilter(img)
#         # img=CLAHE_and_wiener(img)
#         # img=biFilter(img)
#         # img=laplacian(img)
#         # img=biFilter(img)
#         # img=closing(img,31)
#         # img=opening(img)



#         # img=unsharp_musk(img)

#         h,w=img.shape[:2]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         crop_img = img[int(h/8*5):int(h/8*7), int(w/8*3):int(w/8*5)]
#         # crop_img = img[int(h/8*4):int(h/8*8), int(w/8*2):int(w/8*6)]
#         lower_red1 = np.array([0, 40, 100])  # np.array([0, 120, 70])  # np.array([0, 20, 50])  #
#         upper_red1 = np.array([40, 255, 255])  # np.array([10, 255, 255])  # np.array([20, 255, 255])  #
#         lower_red2 = np.array([140, 40, 100])  # np.array([170, 120, 70])  # np.array([150, 20, 50])  #
#         upper_red2 = np.array([180, 255, 255])  # np.array([180, 255, 255])  # np.array([180, 255, 255])  #

#         lower_blue1= np.array([100, 80, 100])
#         upper_blue1 = np.array([130, 255, 255])
#         lower_blue2=np.array([80,100,150])
#         upper_blue2=np.array([100,255,255])
#         mask1 = cv2.inRange(crop_img, lower_red1, upper_red1)
#         mask2 = cv2.inRange(crop_img, lower_red2, upper_red2)
#         mask_red = cv2.bitwise_or(mask1, mask2)
#         mask_blue1 = cv2.inRange(crop_img, lower_blue1, upper_blue1)
#         mask_blue2 = cv2.inRange(crop_img, lower_blue2, upper_blue2)
#         mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)
#         mask = cv2.bitwise_or(mask_red, mask_blue)
#         # cv2.imwrite(fr"E:\2024-2025summer\CV\part\{file_name}", cv2.cvtColor(crop_img,cv2.COLOR_HSV2BGR))
#         # 进行形态学操作以去除噪声
#         kernel = np.ones((15, 15), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 进行开运算（先腐蚀后膨胀）通过判断上边是白色或黑色来翻转图片
#         cv2.imwrite(fr"E:\2024-2025summer\CV\mask2\{name}", mask)

#         # pil_img = Image.open(file_path)
#         # img_draw = ImageDraw.Draw(pil_img)
#         # img_draw = img_draw.rectangle([140, 420, 180, 480], outline= "green", width=5)
#         # pil_img.save(fr"C:\Users\11\Desktop\wwww\111{file_name}")
#         if np.sum(mask) > w/4 * h/4 * 255 * 0.5:  # 检测下面的部分
#             img = cv2.rotate(img, cv2.ROTATE_180)
#             # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
#             # cv2.imwrite(fr"E:\2024-2025summer\CV\rotated\{file_name}", img)
#         # else:
#         #     img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
#         #     cv2.imwrite(fr"E:\2024-2025summer\CV\rotated\{file_name}", img)
#         img=img[int(h/3):]
#         img=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)

#         img=cv2.copyMakeBorder(img,30,30,30,30,cv2.BORDER_CONSTANT,value=(0,90,250))
#         # cv2.imwrite(fr"E:\2024-2025summer\CV\withboard\{file_name}", cv2.cvtColor(img,cv2.COLOR_HSV2BGR))
#         mask1 = cv2.inRange(img, lower_red1, upper_red1)
#         mask2 = cv2.inRange(img, lower_red2, upper_red2)
#         mask_red = cv2.bitwise_or(mask1, mask2)
#         mask_blue1 = cv2.inRange(img, lower_blue1, upper_blue1)
#         mask_blue2 = cv2.inRange(img, lower_blue2, upper_blue2)
#         mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)
#         mask = cv2.bitwise_or(mask_red, mask_blue)
#         kernel = np.ones((15, 15), np.uint8)
#         # cv2.imshow('a',mask)
#         # cv2.waitKey(0)
#         cv2.imwrite(fr"E:\2024-2025summer\CV\NewMasks\{name}", mask)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask=cv2.GaussianBlur(mask,(11,11),100)
#         # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         # mask=cv2.GaussianBlur(mask,(11,11),100)
#         mask=cv2.dilate(mask,(9,9))
#         kernel=np.ones((5,5),np.uint8)
#         mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
#         kernel=np.ones((9,9),np.uint8)
#         mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
#         # cv2.imshow('b',mask)
#         # cv2.waitKey(0)
#         mask[30:110,:]=255
#         mask[590:670,:]=255
#         mask[:,30:110]=255
#         mask[:,590:670]=255
#         cv2.imwrite(fr"E:\2024-2025summer\CV\NewMasks_\{name}", mask)
#         contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         areas= []
#         quad = []
#         # print(contours)
#         for cnt in contours:
#             # 多边形拟合
#             epsilon = 0.05* cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, epsilon, True)
#             hull=cv2.convexHull(approx)
#             epsilon=0.05*cv2.arcLength(hull,True)
#             approx=cv2.approxPolyDP(hull,epsilon,True)
#             # 只保留四边形轮廓
#             if len(approx) == 4 :
#                 area = cv2.contourArea(approx)
#                 if legal_region(area,approx):  # 可以根据实际情况调整最小面积阈值
#                 # if area < min_area and area >45000 and area <610*610:
#                     areas.append(area)
#                     quad.append(approx)
#             elif len(approx)>4:
#                 area = cv2.contourArea(approx)
#                 if legal_region_multi(area,approx.copy()):
#                     rect=cv2.minAreaRect(cnt)
#                     box=cv2.boxPoints(rect)
#                     areas.append(area)
#                     approx=box.reshape((-1,1,2))
#                     quad.append(approx)
#         areas=np.array(areas)
#         quad=np.array(quad)
#         if areas.size !=0:
#             # print(file_name,":")
#             # print(best_quad)
#             if areas.size==1:
#                 points=quad[0].reshape(-1,2)
#                 sum_xy=points.sum(axis=1)
#                 diff_xy=points[:,0]-points[:,1]
#                 corners=np.zeros((4,2),dtype=np.float32)
#                 corners[0]=points[np.argmin(sum_xy)]
#                 corners[1]=points[np.argmax(diff_xy)]
#                 corners[2]=points[np.argmax(sum_xy)]
#                 corners[3]=points[np.argmin(diff_xy)]
#                 corners[0]-=30
#                 corners[2]+=30
#                 corners[1][0]+=30
#                 corners[1][1]-=30
#                 corners[3][0]-=30
#                 corners[3][1]+=30
#                 pts_src=np.float32([
#                     corners[0],
#                     corners[1],
#                     corners[2],
#                     corners[3]
#                 ])
#                 print(name,pts_src)
#                 pts_dest=np.float32([
#                         [0,0],
#                         [639,0],
#                         [639,639],
#                         [0,639],
#                     ]
#                 )
#                 M=cv2.getPerspectiveTransform(pts_src,pts_dest)
#                 img=cv2.warpPerspective(img,M,(640,640),borderValue=(255,255,255),flags=cv2.INTER_LANCZOS4)
#             else:
#                 min1_val = np.min(areas)
#                 min1_idx = np.argmin(areas)

#                 # 把该值临时设置为无穷大
#                 arr_temp = areas.copy()
#                 arr_temp[min1_idx] = np.inf

#                 # 再找第二小的
#                 min2_val = np.min(arr_temp)
#                 min2_idx = np.argmin(arr_temp)
#                 points1=quad[min1_idx].reshape(-1,2)
#                 points2=quad[min2_idx].reshape(-1,2)
#                 points=np.concatenate((points1,points2),axis=0)
#                 # print(name,points)
#                 sum_xy=points.sum(axis=1)
#                 diff_xy=points[:,0]-points[:,1]
#                 corners=np.zeros((4,2),dtype=np.float32)
#                 corners[0]=points[np.argmin(sum_xy)]
#                 corners[1]=points[np.argmax(diff_xy)]
#                 corners[2]=points[np.argmax(sum_xy)]
#                 corners[3]=points[np.argmin(diff_xy)]
#                 corners[0]-=30
#                 corners[2]+=30
#                 corners[1][0]+=30
#                 corners[1][1]-=30
#                 corners[3][0]-=30
#                 corners[3][1]+=30
#                 pts_src=np.float32([
#                     corners[0],
#                     corners[1],
#                     corners[2],
#                     corners[3]
#                 ])
#                 print(name,pts_src)
#                 pts_dest=np.float32([
#                         [0,0],
#                         [639,0],
#                         [639,639],
#                         [0,639],
#                     ]
#                 )
#                 M=cv2.getPerspectiveTransform(pts_src,pts_dest)
#                 img=cv2.warpPerspective(img,M,(640,640),borderValue=(255,255,255),flags=cv2.INTER_LANCZOS4)
#                 print(f"{name}:divi")
#         else:
#             print(name,"no")
#             # print(contours)
#             img=img[110:590,110:590]

#         img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

#         img=biFilter(img)
#         img=CLAHE_and_wiener(img)
#         # img=biFilter(img)
#         img=laplacian(img)
#         # img=biFilter(img)
#         # img=opening(img)
#         img=opening(img,15)
#         img=opening(img,21)
#         img=opening(img,27)
#         img=closing(img,15)
#         img=biFilter(img)
#         h,w=img.shape[:2]
#         img1=img[:,:int(6*w/11)]
#         img2=img[:,int(5*w/11):]
#         img1=cv2.resize(img1,(640,640),interpolation=cv2.INTER_LANCZOS4)
#         img2=cv2.resize(img2,(640,640),interpolation=cv2.INTER_LANCZOS4)
#         cv2.imwrite(os.path.join(path,f"{name[:-4]}_1.jpg"),img1)
#         cv2.imwrite(os.path.join(path,f"{name[:-4]}_2.jpg"),img2)

