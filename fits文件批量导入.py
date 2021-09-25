import matplotlib.pyplot as plt
import numpy
import numpy as np
from astropy.io import fits
import os

num=0

for i in range(2047):
    for j in range(2047):
        x = 0
        for filename in os.listdir(r"D:\yangwangyihao\0618\zw1"):  #listdir的参数是文件夹的路径
            img = fits.open("D:/yangwangyihao/0618/zw1/"+filename)
            num=num+1

            img1 = img[0].data
            arr = numpy.array(img1)
    # mylog = open('数据.data', mode='a', encoding='utf-8')
    # for i in range(len(arr)):
    #     for j in range(len(arr)):
    #         print(arr[i,j],end=" ",file=mylog)
    # print()


    # arr1= numpy.array(img1[0,1])
            arr2 = numpy.array(img1[i, j])
    #
    #         print(x,end=" ")
            mylog = open('shuju.data', mode='a', encoding='utf-8')
    #         print(x, arr2, file=mylog)
    #         x = x + 4
    # print(arr1)
    # print(arr1, end=" ")
            print(arr2,end=" ",file=mylog)
        print(file=mylog)
    # print(arr)
# print(num)                                 #此时的filename是文件夹中文件的名称






# hud = fits.open('D:/yangwangyihao/0618/zw/UV_2021-06-16T17%3A47%3A36.332_4.000s.fits')
# img = hud[0].data
# arr = numpy.array(img)
# # num=np.mean(arr)
# print(arr)
# # plt.imshow(img, cmap= 'gray');
# # plt.colorbar();
# # plt.show();
