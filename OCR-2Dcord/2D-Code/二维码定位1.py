# -*- coding:utf-8 -*-
import numpy as np
import cv2

"""======================图像预处理：转灰度、二值化、形态学膨胀、查找轮廓==================================="""
img_color=cv2.imread(r'./data/c.jpg')#彩图
img_gray=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)#转灰度图
ret,img_ostu=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#二值化
kernel=np.ones((7,7),dtype=np.uint8)#形态学核
img_dilate=cv2.dilate(img_ostu,kernel,iterations=1)#形态学膨胀
cv2.imshow("111",img_dilate)
img_contour,contours,hierarchys=cv2.findContours(img_dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#查找轮廓

"""================轮廓处理，通过轮廓外包矩行，符合一定长宽比的前提下取最大矩形面积即为二维码区域=============="""
Area=0
for i in range(len(contours)):
    x,y,w,h=cv2.boundingRect(contours[i])#轮廓外包矩形
    if((w/h>0.8)&(h/w>0.8)):#符合一定长宽比前提下取最大面积
        if(Area<w*h):
            Area=w*h
            index=i

"""绘制了二维码的外包矩形以及最小外包矩形"""
x,y,w,h=cv2.boundingRect(contours[index])#轮廓外包正矩形
cv2.rectangle(img_color ,(x,y),(x+w,y+h),(0,0,255),2)
rect=cv2.minAreaRect(contours[index])#轮廓最小外包矩形
box=cv2.boxPoints(rect)#矩形点转成四个矩形的四个顶点表示
box=np.uint0(box)
cv2.drawContours(img_color,[box],0,(255,0,0),2)

"""创建只有二维码区域的mask,然后在mask再进行一次查找轮廓，通过层级关系定位出三个定位点，并绘制矩形和连接三个定位点"""
img_mask=np.zeros(img_ostu.shape,np.uint8)
img_mask[y:y+h,x:x+w]=img_ostu[y:y+h,x:x+w]
mask_contour,contours_mask,hierarchys_mask=cv2.findContours(img_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
hierarchys_mask=hierarchys_mask[0]
#筛选出包含一个子层级的轮廓
found=[]
for i in range(len(contours_mask)):
    k=i
    count=0
    while hierarchys_mask[k][2]!=-1:
        k=hierarchys_mask[k][2]
        count+=1
    if count>=2:            #筛选出包含一个子层级的轮廓
       found.append(i)
#为每个定位点绘制最小矩形
rects=[]
for i in found:
    rect=cv2.minAreaRect(contours_mask[i])
    rects.append(rect)
    box=cv2.boxPoints(rect)
    box=np.uint0(box)
    cv2.drawContours(img_color,[box],0,(255,0,0),3)
#用直线连接三个定位点
zuobiao=[]
for i in rects:
    zuobiao.append(i[0])#i[0]为矩形中心点的坐标
zuobiao=np.uint0(zuobiao)
for i in range(len(zuobiao)):
    j=i+1
    while(j<3):
        cv2.line(img_color,tuple(zuobiao[i]),tuple(zuobiao[j]),(255,0,0),5)
        j+=1
#打印出坐标点以及显示mask图像
print("二维码三个定位点的坐标：\n",zuobiao)
cv2.imshow("mask",img_mask)

"""==============================================旋转变换==============================================="""
import math
"""判断二维码三个定位点哪个是中心点zhognxin,左下点zuoxia,右上点youshang"""
zuobiao1=np.float64(zuobiao)#无符号整型计算出现负数会出错
#定位点两两之间的距离
D01=math.sqrt((zuobiao1[0,0]-zuobiao1[1,0])**2+(zuobiao1[0,1]-zuobiao1[1,1])**2)
D02=math.sqrt((zuobiao1[0,0]-zuobiao1[2,0])**2+(zuobiao1[0,1]-zuobiao1[2,1])**2)
D12=math.sqrt((zuobiao1[1,0]-zuobiao1[2,0])**2+(zuobiao1[1,1]-zuobiao1[2,1])**2)
print("三个点位点之间的距离：",D01,D02,D12)
if((D01>=D02)&(D01>=D12)): #距离最大的两个点为左上角和右下角的点
    zhongxin_x=zuobiao[2,0]
    zhongxin_y=zuobiao[2,1]
    if(abs(zhongxin_x-zuobiao1[0,0])>abs(zhongxin_x-zuobiao1[1,0])):        #x坐标与左上角相近的为左下角定位点
        zuoxia_x=zuobiao[1,0]
        zuoxia_y=zuobiao[1,1]
        youshang_x=zuobiao[0,0]
        youshang_y=zuobiao[0,1]
    else:
        zuoxia_x=zuobiao[0,0]
        zuoxia_y=zuobiao[0,1]
        youshang_x=zuobiao[1,0]
        youshang_y=zuobiao[1,1]
if((D02>=D01)&(D02>=D12)):
    zhongxin_x=zuobiao[1,0]
    zhongxin_y=zuobiao[1,1]
    if(abs(zhongxin_x-zuobiao1[0,0])>abs(zhongxin_x-zuobiao1[2,0])):
        zuoxia_x=zuobiao[2,0]
        zuoxia_y=zuobiao[2,1]
        youshang_x=zuobiao[0,0]
        youshang_y=zuobiao[0,1]
    else:
        zuoxia_x=zuobiao[0,0]
        zuoxia_y=zuobiao[0,1]
        youshang_x=zuobiao[2,0]
        youshang_y=zuobiao[2,1]
if((D12>=D01)&(D12>=D02)):
    zhongxin_x=zuobiao[0,0]
    zhongxin_y=zuobiao[0,1]
    if (abs(zhongxin_x - zuobiao1[1, 0]) > abs(zhongxin_x - zuobiao1[2, 0])):
        zuoxia_x = zuobiao[2, 0]
        zuoxia_y = zuobiao[2, 1]
        youshang_x = zuobiao[1, 0]
        youshang_y = zuobiao[1, 1]
    else:
        zuoxia_x = zuobiao[1, 0]
        zuoxia_y = zuobiao[1, 1]
        youshang_x = zuobiao[2, 0]
        youshang_y = zuobiao[2, 1]
#红圆圈显示左上角点，绿圆圈显示左下角的点
cv2.circle(img_color,(zhongxin_x,zhongxin_y),10,(0,0,255),5)
cv2.circle(img_color,(zuoxia_x,zuoxia_y),10,(0,255,0),5)
cv2.imshow("final",img_color)

"""判断是否是否需要旋转180度"""
if(zuoxia_y<zhongxin_y):
    Rotate180=180
else:
    Rotate180=0
"""计算旋转的角度"""
X=np.float64(zuoxia_x)-zhongxin_x
Y=np.float64(zuoxia_y)-zhongxin_y
angle=math.atan(Y/X)*180/math.pi    #超过九十度是会自动减去180°
if angle<0:
    angle=180+angle
print('Rotate_angle:',angle)
print("左下角",zuoxia_x,zuoxia_y)
print("左上角",zhongxin_x,zhongxin_y)
print("右下角",youshang_x,youshang_y)
"""计算旋转矩阵并进行旋转"""
M1=cv2.getRotationMatrix2D(((img_color.shape[1]//2),(img_color.shape[0]//2)),angle-90+Rotate180,1)#此处的角度是逆时针旋转，如果为负数则为顺时针旋转
img_rotation=cv2.warpAffine(img_color,M1,(img_color.shape[1],img_color.shape[0]))
"""在旋转图中显示原来三个定位点的位置"""
"""
for i in found:
    rect=cv2.minAreaRect(contours_mask[i])
    rects.append(rect)
    box=cv2.boxPoints(rect)
    box=np.uint0(box)
    cv2.drawContours(img_rotation,[box],0,(255,255,0),3)
"""
"""计算原三个定位点旋转之后的坐标"""
sita=90-angle-Rotate180 #此处是旋转角度的相反数，原因没弄明白
centerx=img_color.shape[1]//2
centery=img_color.shape[0]//2
new_zhongxin_x=math.cos(sita*math.pi/180)*zhongxin_x-math.sin(sita*math.pi/180)*zhongxin_y+(1-math.cos(sita*math.pi/180))*centerx+math.sin(sita*math.pi/180)*centery
new_zhongxin_y=math.sin(sita*math.pi/180)*zhongxin_x+math.cos(sita*math.pi/180)*zhongxin_y+(1-math.cos(sita*math.pi/180))*centery-math.sin(sita*math.pi/180)*centerx
cv2.circle(img_rotation,(int(new_zhongxin_x),int(new_zhongxin_y)),20,(0,0,0),10)
new_zuoxia_x=math.cos(sita*math.pi/180)*zuoxia_x-math.sin(sita*math.pi/180)*zuoxia_y+(1-math.cos(sita*math.pi/180))*centerx+math.sin(sita*math.pi/180)*centery
new_zuoxia_y=math.sin(sita*math.pi/180)*zuoxia_x+math.cos(sita*math.pi/180)*zuoxia_y+(1-math.cos(sita*math.pi/180))*centery-math.sin(sita*math.pi/180)*centerx
cv2.circle(img_rotation,(int(new_zhongxin_x),int(new_zuoxia_y)),20,(0,0,0),10)
new_youshang_x=math.cos(sita*math.pi/180)*youshang_x-math.sin(sita*math.pi/180)*youshang_y+(1-math.cos(sita*math.pi/180))*centerx+math.sin(sita*math.pi/180)*centery
new_youshang_y=math.sin(sita*math.pi/180)*youshang_x+math.cos(sita*math.pi/180)*youshang_y+(1-math.cos(sita*math.pi/180))*centery-math.sin(sita*math.pi/180)*centerx
cv2.circle(img_rotation,(int(new_youshang_x),int(new_youshang_y)),20,(0,0,0),10)

"""==============================用矩形框选出各个重要信息的位置==============================="""
"""计算实际一毫米在图像中占据多少像素"""
distance=math.sqrt((new_zuoxia_x-new_zhongxin_x)**2+(new_zuoxia_y-new_zhongxin_y)**2)
print("左上角到左下角的距离：",distance)
dst_permm=distance/11.5 #每毫米多少像素
img_new=np.zeros(img_color.shape,np.uint8)
y=0
#始发站
shizhan_1x=int(new_zhongxin_x-dst_permm*63)
shizhan_1y=int(new_zhongxin_y-dst_permm*28)
shizhan_2x=int(new_zhongxin_x-dst_permm*37)
shizhan_2y=int(new_zhongxin_y-dst_permm*22)

img_new[shizhan_1y-shizhan_1y:shizhan_2y-shizhan_1y,shizhan_1x-shizhan_1x:shizhan_2x-shizhan_1x]=img_rotation[shizhan_1y:shizhan_2y,shizhan_1x:shizhan_2x]
y+=shizhan_2y-shizhan_1y
print("shizhan:",shizhan_2y)
cv2.rectangle(img_rotation,(shizhan_1x,shizhan_1y),(shizhan_2x,shizhan_2y),(0,255,255),3)
#车次
checi_1x=int(new_zhongxin_x-dst_permm*35)
checi_1y=int(new_zhongxin_y-dst_permm*28)
checi_2x=int(new_zhongxin_x-dst_permm*17)
checi_2y=int(new_zhongxin_y-dst_permm*22)

img_new[y:y+checi_2y-checi_1y,0:checi_2x-checi_1x]=img_rotation[checi_1y:checi_2y,checi_1x:checi_2x]
y+=checi_2y-checi_1y
cv2.rectangle(img_rotation,(checi_1x,checi_1y),(checi_2x,checi_2y),(0,0,255),3)
#终点站
zhongzhan_1x=int(new_zhongxin_x-dst_permm*15)
zhongzhan_1y=int(new_zhongxin_y-dst_permm*28)
zhongzhan_2x=int(new_zhongxin_x+dst_permm*10)
zhongzhan_2y=int(new_zhongxin_y-dst_permm*22)

img_new[y:y+zhongzhan_2y-zhongzhan_1y,0:zhongzhan_2x-zhongzhan_1x]=img_rotation[zhongzhan_1y:zhongzhan_2y,zhongzhan_1x:zhongzhan_2x]
y+=zhongzhan_2y-zhongzhan_1y
cv2.rectangle(img_rotation,(zhongzhan_1x,zhongzhan_1y),(zhongzhan_2x,zhongzhan_2y),(0,0,255),3)
#时间
time_1x=int(new_zhongxin_x-dst_permm*65)
time_1y=int(new_zhongxin_y-dst_permm*18)
time_2x=int(new_zhongxin_x-dst_permm*21)
time_2y=int(new_zhongxin_y-dst_permm*13.5)

img_new[y:y+time_2y-time_1y,0:time_2x-time_1x]=img_rotation[time_1y:time_2y,time_1x:time_2x]
y+=time_2y-time_1y
cv2.rectangle(img_rotation,(time_1x,time_1y),(time_2x,time_2y),(0,0,255),3)
#座位
zuowei_1x=int(new_zhongxin_x-dst_permm*20)
zuowei_1y=int(new_zhongxin_y-dst_permm*18)
zuowei_2x=int(new_zhongxin_x+dst_permm*6)
zuowei_2y=int(new_zhongxin_y-dst_permm*13.5)

img_new[y:y+zuowei_2y-zuowei_1y,0:zuowei_2x-zuowei_1x]=img_rotation[zuowei_1y:zuowei_2y,zuowei_1x:zuowei_2x]
y+=zuowei_2y-zuowei_1y
cv2.rectangle(img_rotation,(zuowei_1x,zuowei_1y),(zuowei_2x,zuowei_2y),(0,0,255),3)
#票价
price_1x=int(new_zhongxin_x-dst_permm*65)
price_1y=int(new_zhongxin_y-dst_permm*13.5)
price_2x=int(new_zhongxin_x-dst_permm*45)
price_2y=int(new_zhongxin_y-dst_permm*8.5)

img_new[y:y+price_2y-price_1y,0:price_2x-price_1x]=img_rotation[price_1y:price_2y,price_1x:price_2x]
y+=price_2y-price_1y
cv2.rectangle(img_rotation,(price_1x,price_1y),(price_2x,price_2y),(0,0,255),3)
#座位类型
seattype_1x=int(new_zhongxin_x-dst_permm*20)
seattype_1y=int(new_zhongxin_y-dst_permm*14)
seattype_2x=int(new_zhongxin_x+dst_permm*10)
seattype_2y=int(new_zhongxin_y-dst_permm*8.5)

img_new[y:y+seattype_2y-seattype_1y,0:seattype_2x-seattype_1x]=img_rotation[seattype_1y:seattype_2y,seattype_1x:seattype_2x]
y+=seattype_2y-seattype_1y
cv2.rectangle(img_rotation,(seattype_1x,seattype_1y),(seattype_2x,seattype_2y),(0,0,255),3)
#身份证
ID_1x=int(new_zhongxin_x-dst_permm*65)
ID_1y=int(new_zhongxin_y-dst_permm*0)
ID_2x=int(new_zhongxin_x-dst_permm*27.5)
ID_2y=int(new_zhongxin_y+dst_permm*4)

img_new[y:y+ID_2y-ID_1y,0:ID_2x-ID_1x]=img_rotation[ID_1y:ID_2y,ID_1x:ID_2x]
y+=ID_2y-ID_1y
cv2.rectangle(img_rotation,(ID_1x,ID_1y),(ID_2x,ID_2y),(0,0,255),3)
#名字
Name_1x=int(new_zhongxin_x-dst_permm*27)
Name_1y=int(new_zhongxin_y-dst_permm*0)
Name_2x=int(new_zhongxin_x-dst_permm*10)
Name_2y=int(new_zhongxin_y+dst_permm*4)

img_new[y:y+Name_2y-Name_1y,0:Name_2x-Name_1x]=img_rotation[Name_1y:Name_2y,Name_1x:Name_2x]
y+=Name_2y-Name_1y
cv2.rectangle(img_rotation,(Name_1x,Name_1y),(Name_2x,Name_2y),(0,0,255),3)


cv2.imshow("new",img_new)
cv2.imshow('rotate1',img_rotation)
cv2.imwrite("result7.jpg",img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
