# %%
import cv2
import numpy as np


# %% 显示图片
def show(o):
    cv2.imshow("show", o)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
lena = cv2.imread("resources/lena.jpg")

# %% 转灰度
lena_gray = cv2.imread("resources/street_2.jpg", cv2.IMREAD_GRAYSCALE)

#%%
lena_edge_1 = cv2.Canny(lena_gray,80,150)
lena_edge_2 = cv2.Canny(lena_gray,50,100)
lena_edge_12 = np.hstack((lena_edge_1,lena_edge_2))
show(lena_edge_12)
