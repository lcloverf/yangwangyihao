import matplotlib.pyplot as plt
import numpy
import numpy as np
from astropy.io import fits
import os
hud = fits.open('D:/yangwangyihao/0618/zw1/2.fits')
img = hud[0].data
arr = numpy.array(img)
# mylog = open('gmm', mode='a', encoding='utf-8')
# num=np.mean(arr)
# print( arr, file=mylog)
print(arr.shape)
# plt.imshow(img, cmap= 'gray');
# plt.colorbar();
# plt.show();
