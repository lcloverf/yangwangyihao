import os
import re
import numpy as np
from astropy.io import fits
import SEx1


def Expert_Fits_File(filename,image,mask1):
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename)
    image = image - mask1


def Select_Temperature(mask,image):
    MSE = (np.square(np.multiply(mask, image))).mean()
    print(MSE)
    num = 0
    time = 4
    with fits.open('2.fits') as hdu:
        img1 = hdu[0].data
    MSE1 = (np.square(np.multiply(mask, img1))).mean()
    select1 = abs(MSE1 - MSE)
    for filename1 in os.listdir(r"/home/lab30202/sdc/lvchao/zw1"):  # listdir的参数是文件夹的路径
        img = fits.open("/home/lab30202/sdc/lvchao/zw1/" + filename1)
        img2 = img[0].data
        img2 = np.array(img2)
        num = num + 1
        MSE2 = (np.square(np.multiply(mask, img1))).mean()
        select2 = abs(MSE2 - MSE)
        if (select1 > select2):
            select1 = select2
            time = num * 4
    return time

def Generate_Dark_Current_Matrix(time):
    mask1 = np.loadtxt('20211116150236mask2.npy')
    func = np.loadtxt('20211116150236f1_arr.npy')
    for i in range(func.shape[1]):
        b = func[i]
        x = time
        y = b[0] * pow(x, 10) + b[1] * pow(x, 9) + b[2] * pow(x, 8) + b[3] * pow(x, 7) + b[4] * pow(x, 6) + b[
            5] * pow(x, 5) + b[6] * pow(x, 4) + b[7] * pow(x, 3) + b[8] * pow(x, 2) + b[9] * pow(x, 1) + b[10]
        num = i  # 想要替换的数字
        NUM = y  # 替换后的数字
        index = (mask1 == num)
        mask1[index] = NUM
    mask1[np.isnan(mask1)] = 0
    return mask1

def Generate_Mask_Matrix(result):
    mask = np.ones((2048, 2048))
    for i in range(result.shape[1]):
        a = result[i]
        mask[a[0] - 5:a[0] + 5, a[1] + 5:a[1] - 5] = 0  # 截取一块矩阵
    x, y = [], []
    with open('outlier.data') as A:
        for eachline in A:
            tmp = re.split("\s+", eachline.rstrip())
            x.append(tmp[0])
            y.append(tmp[1])
    for i in range(len(x)):
        mask[int(x[i]), int(y[i])] = 0
    return mask


def Generate_Galaxy_Coordinates(fitPath):
    keys = ['DETECT_TYPE',
            'DETECT_MINAREA',
            'DETECT_THRESH',
            'ANALYSIS_THRESH',
            'DEBLEND_NTHRESH',
            'DEBLEND_MINCONT',
            'CLEAN_PARAM',
            'BACK_SIZE',
            'BACK_FILTERSIZE']

    values = ['CCD', 18, 5.0, 5.0, 64, 0.065, 1.0, 100, 3]
    sex = SEx1.ColdHotdetector(
        fitPath, 'image2.sex', 'simplify.sex', 'test.param', True, '741236985',
        keys, values)
    result = sex.cold
    # print(type(result))
    result = [i[:2] for i in result]
    result = np.array(result, dtype='int_')
    print(result)
    # np.savetxt(name+".npy", result,fmt='%d')
    image = sex.read_fits()
    return result,image


def Remove_Dark(Ori_Datas_path):
    for filename in os.listdir(Ori_Datas_path):  # listdir的参数是文件夹的路径
        fitPath = Ori_Datas_path+"/" + filename
        result,image=Generate_Galaxy_Coordinates(fitPath)
        mask=Generate_Mask_Matrix(result)
        time=Select_Temperature(mask,image)
        mask1=Generate_Dark_Current_Matrix(time)
        Expert_Fits_File(filename,image,mask1)
if __name__ == '__main__':
    Remove_Dark("/home/lab30202/sdc/YangWang_1/UV/datas/Ori_Datas/fits")



