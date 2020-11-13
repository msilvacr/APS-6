# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:59:30 2020

@author: marlon.cruz
"""

import cv2
# import os
# import sys
import numpy
import matplotlib.pyplot as plt
from modulos import image_enhance
# from skimage.morphology import skeletonize, thin

# 
# os.chdir("/app/")
# 

def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)

    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2


def get_descriptors(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy)
	
    
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	
    
	img[img == 255] = 1

	
	skeleton = cv2.skeletonize(img)
	skeleton = numpy.array(skeleton, dtype=numpy.uint8)
	skeleton = removedot(skeleton)
    
    
	harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125
	# Extrair keypoints
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))
	# Define descriptor
	orb = cv2.ORB_create()
	
	_, des = orb.compute(img, keypoints)
	return (keypoints, des);


def main():
    
    img1 = '101_1.tif'
    
    #nome_img = sys.argv[1]
    imagem1 = cv2.imread("repos_img/" + img1, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = get_descriptors(imagem1)


    img2 = '102_1.tif'
# 	nome_img = sys.argv[2]
    imagem2 = cv2.imread("database/" + img2, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = get_descriptors(imagem2)


    # Realizar match entre as descrições 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
	# Plot keypoints
    img4 = cv2.drawKeypoints(imagem1, kp1, outImage=None)
    img5 = cv2.drawKeypoints(imagem2, kp2, outImage=None)
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img4)
    axarr[1].imshow(img5)
    plt.show()
	# Plot matches
    imagem3 = cv2.drawMatches(imagem1, kp1, imagem2, kp2, matches, flags=2, outImg=None)
    plt.imshow(imagem3)
    plt.show()

# 	verificar score de match
    score = 0;
    for match in matches:
        score += match.distance
    score_threshold = 33
    if score/len(matches) < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")


if __name__ == "__main__":
	try:
		main()
	except:
		raise
