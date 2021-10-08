import matplotlib.pyplot as plt
import numpy
import time
import numpy as np
import pandas as pd
from astropy.io import fits
import os
import numpy.ma as npm
from sklearn import mixture
from sklearn.mixture import GaussianMixture

num=0

com1=np.empty((1,2048,2048))

hud1 = fits.open('D:/yangwangyihao/0618/zw2/1.fits')
img1 = hud1[0].data
arr1 = numpy.array(img1)
for filename in os.listdir(r"D:\yangwangyihao\0618\zw1"):  #listdir的参数是文件夹的路径
    img = fits.open("D:/yangwangyihao/0618/zw1/"+filename)
    num=num+1

    img1 = img[0].data
    arr = np.array(img1)
    arr=arr-arr1
    # print('维数：', arr.shape)
    # print(arr)
    com=np.array([arr])
    com1=np.concatenate((com1,com))
com2=numpy.delete(com1, 0, axis=0)
print(num)
print('维数：',com2.shape)
# mylog = open('1', mode='a', encoding='utf-8')
# print(com2)
print("*" * 30)
X=com2[:, :60, : 60]
old_shape =X.shape
X = X.reshape(-1,35)
print(X)
print((X.shape))
print("*"*30)
# X=com2[:, 60,  60]
# print(X)
# print(X.shape)
# print(com2[:, 2:, : 1])

# print(com2[0,0,0])


def aic_bic(x):  # 得到最优聚类数目
    x = x.reshape(-1,35)
    n_components = np.arange(1, 15)
    start1_time=time.time()
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(x)
              for n in n_components ]
    start_time=time.time()
    plt.plot(n_components, [m.bic(x) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(x) for m in models], label='AIC')
    end_time=time.time()
    print("一次运行的时间：",end_time-start_time)
    plt.legend(loc='best')
    plt.xlabel('n_components')
    # plt.savefig('scatter.jpg')#保存图片
    plt.show()



# n_components=3


def gaussian_mixture(x,n):  # 得到最初的聚类数组
    gmm = mixture.GaussianMixture(covariance_type='full', n_components=n)
    gmm.fit(x)
    clusters = gmm.predict(x)
    # print(clusters[0])
    clusters = clusters.reshape(old_shape[2], old_shape[1])
    mask=clusters
    mask = np.array(mask, dtype=float)
    return mask





def judge_outliers(n,mask):# 判断异常点，并把异常点置None，删除异常点坐标对应的X中的一行数据

    for g in range(n):
        com3 = np.empty((1, 35))
        xx = []
        yy = []
        for i in range(len(mask)):
            for j in range(len(mask)):
                if (mask[i][j] == g):
                    com4 = com2[:, i, j]
                    xx.append(i)
                    yy.append(j)
                    # num=num+1
                    # print(com4.shape)
                    com3 = np.vstack((com3, com4))
        # print(num)
        #     print(xx)
        #     print(yy)
        com3 = numpy.delete(com3, 0, axis=0)
        print(com3.shape)
        # print(com3.mean(axis=0))#列

        x = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104,
             108, 112, 116, 120, 124, 128, 132, 136]
        x = np.array(x)
        # print('x is :\n', x)
        y = com3.mean(axis=0)
        y = np.array(y)
        # print('y is :\n', y)
        # 用5次多项式拟合
        f1 = np.polyfit(x, y, 10)
        # print('f1 is :\n', f1)

        p1 = np.poly1d(f1)
        # print('p1 is :\n', p1)

        # 也可使用yvals=np.polyval(f1, x)
        yvals = p1(x)  # 拟合y值
        # print('yvals is :\n', yvals)
        # mse_num=np.maximum((yvals-com3),-(yvals-com3)).sum()
        B = numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35
        # print(numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35)
        mean = np.mean(numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35)  # 求均值
        print("均值：", mean)
        var = np.var(numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35)  # 求方差
        print("方差：", var)

        for h in range(len(B)):  # 判断异常点
            if B[h] > (3* mean):
                mylog = open('outlier.data', mode='a', encoding='utf-8')
                print(xx[h], yy[h],file=mylog)
                mask[xx[h], yy[h]] = None
                index = yy[h] * 60 + xx[h]
                global X
                X = np.delete(X, index, axis=0)

    return mask,X

def gauss_clustering_again(x,n,mask):  # 屏蔽异常点后在聚类
    gmm = mixture.GaussianMixture(covariance_type='full', n_components=n)
    gmm.fit(x)
    clusters = gmm.predict(x)
    # print(clusters[0])
    # clusters = clusters.reshape(old_shape[2], old_shape[1])  # 根据mask的index赋值，先循环，
    # com3=np.empty((1,36))
    h = 0
    happened = False
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            if str(mask[i, j]) == 'nan':
                mask[i, j] = float('nan')
            else:
                mask[i, j] = clusters[h]
                h = h + 1
    for g in range(n):
        com3 = np.empty((1, 35))
        xx = []
        yy = []
        for i in range(len(mask)):
            for j in range(len(mask)):
                if (mask[i][j] == g):
                    com4 = com2[:, i, j]
                    xx.append(i)
                    yy.append(j)
                    # num=num+1
                    # print(com4.shape)
                    com3 = np.vstack((com3, com4))
        # print(num)
        #     print(xx)
        #     print(yy)
        com3 = numpy.delete(com3, 0, axis=0)
        print(com3.shape)
        # print(com3.mean(axis=0))#列

        x = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104,
             108, 112, 116, 120, 124, 128, 132, 136]
        x = np.array(x)
        # print('x is :\n', x)
        y = com3.mean(axis=0)
        y = np.array(y)
        # print('y is :\n', y)
        # 用5次多项式拟合
        f1 = np.polyfit(x, y, 10)
        # print('f1 is :\n', f1)

        p1 = np.poly1d(f1)
        # print('p1 is :\n', p1)

        # 也可使用yvals=np.polyval(f1, x)
        yvals = p1(x)  # 拟合y值
        # print('yvals is :\n', yvals)
        # mse_num=np.maximum((yvals-com3),-(yvals-com3)).sum()
        B = numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35
        print(numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35)
        mean = np.mean(numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35)  # 求均值
        print("均值：", mean)
        var = np.var(numpy.sqrt((numpy.square(yvals - com3).sum(axis=1))) / 35)  # 求方差
        print("方差：", var)

        for h in range(len(B)):  # 判断异常点
            if B[h] > (3 * mean):
                happened = True
                mylog = open('outlier.data', mode='a', encoding='utf-8')
                print(xx[h], yy[h],file=mylog)

                mask[xx[h], yy[h]] = None
                index = yy[h] * 60 + xx[h]
                global X
                X = np.delete(X, index, axis=0)

    return mask,happened

aic_bic(X)
# mask=gaussian_mixture(X,6)
# mask,X= judge_outliers(6,mask)
#
# # X=judge_outliers(6,mask)[1]
# print('*'*30)
# mask,signal=gauss_clustering_again(X,6,mask)
# num1=0
# while(signal):
#     print("*"*30)
#     mask, signal = gauss_clustering_again(X, 6, mask)
#     print("X矩阵形状：",X.shape)
#     num1=num1+1
# print(num1)

# mask1=gaussian_mixture(X,6)

