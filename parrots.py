import numpy as np
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte
import pylab
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math


def PSNR(im, im_new):
    size = im.shape
    cord_1, cord_2, cord_3 = 0, 0, 0
    for i in range(size[0]):
        for k in range(size[1]):
            cord_1 += (im[i][k][0] - im_new[i][k][0])**2
            cord_2 += (im[i][k][1] - im_new[i][k][1])**2
            cord_3 += (im[i][k][2] - im_new[i][k][2])**2
    MSE = 1/(3*size[0]*size[1])*(cord_1 + cord_2 + cord_3)

    PSNR = 20*math.log10(1) - 10*math.log10(MSE)
    return PSNR


            
image = imread('parrots.jpg')
size = image.shape 
im = pylab.imshow(image)
pylab.show()

image = img_as_float(image)


X = np.reshape(image, (size[0]* size[1], 3))

for k in range(9,10):
    model = KMeans(n_clusters = k , init ='k-means++',random_state = 241)
    model.fit(X)

    labels = model.labels_
    centres = model.cluster_centers_
    print(labels)

    mean_pix = []
    med_pix = []
    for i in range(k):
        inds = list(*np.where(labels == i))
        mean, med = [0,0,0] , [[],[],[]]
        for d in range(len(inds)):
            mean[0]+=X[inds[d]][0]
            mean[1]+=X[inds[d]][1]
            mean[2]+=X[inds[d]][2]
            med[0].append(X[inds[d]][0])
            med[1].append(X[inds[d]][1])
            med[2].append(X[inds[d]][2])
        mean = np.array(mean)/len(inds)
        med[0] = np.median(med[0])
        med[1] = np.median(med[1])
        med[2] = np.median(med[2])
        
        mean_pix.append(mean)
        med_pix.append(med)


    image_med = np.array(list(map( lambda pix: med_pix[pix] , labels)))
    image_mean = np.array(list(map( lambda pix: mean_pix[pix] , labels)))
    image_centr =  np.array(list(map( lambda pix: centres[pix] , labels)))

    image_mean = np.reshape(image_mean, (size[0], size[1], 3))
    image_med = np.reshape(image_med, (size[0], size[1], 3))
    image_centr = np.reshape(image_centr, (size[0], size[1], 3))
    

  
    im1 = pylab.imshow(image_mean)
    pylab.show()
    im2 = pylab.imshow(image_med)
    pylab.show()
    im2 = pylab.imshow(image_centr)
    pylab.show()
    print("K",k,"PSNR MEAN:",PSNR(image,image_mean),"MED",PSNR(image,image_med),"Centr",PSNR(image,image_centr))
