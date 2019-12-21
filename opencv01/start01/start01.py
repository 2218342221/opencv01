# %%
import cv2


# %% 显示图片
def show(o):
    cv2.imshow("show", o)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
lena = cv2.imread("resources/lena.jpg")

# %%
cv2.imshow("lena", lena)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 转灰度
lena_gray = cv2.imread("resources/lena.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("lena_gray", lena_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 保存
cv2.imwrite("resources/lena_gray.jpg", lena_gray)

# 视频读取 cv2.VideoCapture() 可以捕捉摄像头，也可以指定路径，读视频
# %%
wild_life = cv2.VideoCapture("resources/Wildlife.wmv")
if wild_life.isOpened():
    while True:
        ret, frame = wild_life.read()
        if frame is None:
            break
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("result", gray)
            if cv2.waitKey(100) & 0xFF == 27:
                break
wild_life.release()
cv2.destroyAllWindows()

# %% 切割和 b g r 分离
show(lena[0:100, 0:200, :])
b, g, r = cv2.split(lena)
show(b)  # lena[:,:,0]
show(g)
show(r)
# %%
show(cv2.merge((b, g, r)))

# %% 边界填充
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

replication = cv2.copyMakeBorder(lena, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(lena, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(lena, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(lena, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(lena, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                              value=0)
import matplotlib.pyplot as plt

plt.subplot(231)
plt.imshow(lena), plt.title("origin")
plt.subplot(232)
plt.imshow(replication), plt.title("replication")
plt.subplot(233)
plt.imshow(reflect), plt.title("reflect")
plt.subplot(234)
plt.imshow(reflect101), plt.title("reflect101")
plt.subplot(235)
plt.imshow(wrap), plt.title("wrap")
plt.subplot(236)
plt.imshow(constant), plt.title("constant")
plt.show()

# %% cv2.add(lena,lena1)  超过255就算255

# %%
lena2 = cv2.resize(lena, (0, 0), fx=1.5, fy=1)
show(lena2)
lena3 = cv2.resize(lena, (150, 220))  # lena3.shape is (220, 150, 3)  150:width  220:height
show(lena3)

# %% cv2.addWeighted(pic1,0.4,pic2,0.6,0)

# %%阈值
'''
cv2.threshold(img_gray,thresh,maxval,cv2.THRESH_BINARY)  大于阈值的取maxval 否则取零
cv2.threshold(img_gray,thresh,maxval,cv2.THRESH_BINARY_INV)  小于阈值取maxval,否则取零
cv2.threshold(img_gray,thresh,maxval,cv2.THRESH_TRUNC) 大于阈值取阈值
cv2.threshold(img_gray,thresh,maxval,cv2.THRESH_TOZERO) 小于阈值取0
cv2.threshold(img_gray,thresh,maxval,cv2.THRESH_TOZERO_INV)  大于阈值取0
'''

# %%滤波
lena_noise = cv2.imread("resources/lena_noise.jpg")
show(lena_noise)
# 均值滤波
lena_noise_blur = cv2.blur(lena_noise, (5, 5))  # 5*5内取均值
show(lena_noise_blur)
lena_noise_box = cv2.boxFilter(lena_noise, -1, (3, 3), normalize=True)
show(lena_noise_box)
# 高斯滤波
lena_noise_gaussian = cv2.GaussianBlur(lena_noise, (5, 5), 3)
show(lena_noise_gaussian)
# 中值滤波
lena_noise_median = cv2.medianBlur(lena_noise, 5)
show(lena_noise_median)
import numpy as np

lena_noises = np.hstack((lena_noise, lena_noise_blur, lena_noise_box, lena_noise_gaussian, lena_noise_median))
show(lena_noises)

# %% 腐蚀操作
kernal = np.ones((3, 3), np.uint8)
lena_erode = cv2.erode(lena, kernal, iterations=1)
show(lena_erode)

# %% 膨胀
lena_dilate = cv2.dilate(lena, kernal, iterations=1)

# %% 形态学计算
'''
开:先腐蚀,再膨胀 
闭：先膨胀，再腐蚀  
梯度：膨胀-腐蚀
礼帽：原始输入-开运算
黑帽：闭运算-原始输入
'''

lena_open = cv2.morphologyEx(lena, cv2.MORPH_OPEN, kernal)
lena_close = cv2.morphologyEx(lena, cv2.MORPH_CLOSE, kernal)
lena_gradient = cv2.morphologyEx(lena, cv2.MORPH_GRADIENT, kernal)
show(lena_gradient)
lena_blackhat = cv2.morphologyEx(lena, cv2.MORPH_BLACKHAT, kernal)
lena_tophat = cv2.morphologyEx(lena, cv2.MORPH_TOPHAT, kernal)
show(lena_blackhat)
show(lena_tophat)

# %% sobel  建议dx dy不要都赋值1， 分开计算再相加。
lena_sobel = cv2.Sobel(lena, cv2.CV_64F, 1, 1, ksize=3)  # cv2.CV_64F 负数浮点 1,0 只有dx
lena_sobel_abs = cv2.convertScaleAbs(lena_sobel)  # 计算绝对值
show(lena_sobel)
show(lena_sobel_abs)

#%%
lena_sobel_x = cv2.Sobel(lena, cv2.CV_64F, 1, 0, ksize=3)  # cv2.CV_64F 负数浮点 1,0 只有dx
lena_sobel_y = cv2.Sobel(lena, cv2.CV_64F, 0, 1, ksize=3)  # cv2.CV_64F 负数浮点 1,0 只有dx
lena_sobel_abs_x = cv2.convertScaleAbs(lena_sobel_x)  # 计算绝对值
lena_sobel_abs_y = cv2.convertScaleAbs(lena_sobel_y)
lena_sobel_abs_xy = cv2.addWeighted(lena_sobel_abs_x,0.5,lena_sobel_abs_y,0.5,0)
lena_sobel_abs_xy1 = cv2.threshold(lena_sobel_abs_x.astype(np.int16)+lena_sobel_abs_y,255,255,cv2.THRESH_TRUNC)
show(lena_sobel_abs_xy)
show(lena_sobel_abs_xy1[1].astype(np.int8))


#%% scharr算子 laplacian算子
lena_scharr_x = cv2.Scharr(lena_gray, cv2.CV_64F, 1, 0)  # cv2.CV_64F 负数浮点 1,0 只有dx
lena_scharr_y = cv2.Scharr(lena_gray, cv2.CV_64F, 0, 1)  # cv2.CV_64F 负数浮点 1,0 只有dx
lena_scharr_abs_x = cv2.convertScaleAbs(lena_scharr_x)  # 计算绝对值
lena_scharr_abs_y = cv2.convertScaleAbs(lena_scharr_y)  # 计算绝对值
show(cv2.addWeighted(lena_scharr_abs_x,0.5,lena_scharr_abs_y,0.5,0))

lena_laplacian = cv2.Laplacian(lena_gray, cv2.CV_64F)
lena_laplacian_abs = cv2.convertScaleAbs(lena_laplacian)
show(lena_laplacian_abs)