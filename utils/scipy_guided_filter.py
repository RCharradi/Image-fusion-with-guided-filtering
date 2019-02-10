# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:29:16 2018

@author: Ramzi Charradi
"""
from scipy.ndimage.filters import laplace
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')


# Algorithm using scipy's guided Filter

def fusion(im1,im2):
    sigma_r = 5
    average_filter_size=31
    r_1=45
    r_2=7
    eps_1=0.3
    eps_2=10e-6
    

    if im1.max()>1:
        im1=im1/255
    if im2.max()>1:
        im2=im2/255
            
    im1_blue, im1_green, im1_red = cv2.split(im1)
    im2_blue, im2_green, im2_red = cv2.split(im2)
            
    base_layer1 = uniform_filter(im1, mode='reflect',size=average_filter_size)
    b1_blue, b1_green, b1_red = cv2.split(base_layer1)
            
    base_layer2 = uniform_filter(im2, mode='reflect',size=average_filter_size)
    b2_blue, b2_green, b2_red = cv2.split(base_layer2)
            
    detail_layer1 = im1 - base_layer1
    d1_blue, d1_green, d1_red = cv2.split(detail_layer1)
            
    detail_layer2 = im2 - base_layer2
    d2_blue, d2_green, d2_red = cv2.split(detail_layer2)
            
            
    saliency1 = gaussian_filter(abs(laplace(im1_red+im1_green+im1_blue,mode='reflect')),sigma_r,mode='reflect')
    saliency2 = gaussian_filter(abs(laplace(im2_red+im2_green+im2_blue,mode='reflect')),sigma_r,mode='reflect')
    mask = np.float32(np.argmax([saliency1, saliency2], axis=0))
    
    im1=np.float32(im1)
    im2=np.float32(im2)
    
    gf1 = cv2.ximgproc.createGuidedFilter(im1, r_1, eps_1)
    gf2 = cv2.ximgproc.createGuidedFilter(im2, r_1, eps_1)  
    gf3 = cv2.ximgproc.createGuidedFilter(im1, r_2, eps_2)
    gf4 = cv2.ximgproc.createGuidedFilter(im2, r_2, eps_2)
        
    g1r1 = gf1.filter(1 - mask)
    g2r1 = gf2.filter(mask)
    g1r2 = gf3.filter(1-mask)
    g2r2 = gf4.filter(mask)
    
    fused_base1 = np.float32((b1_blue * (g1r1) + b2_blue * (g2r1))/((g1r1+g2r1)))          
    fused_detail1 = np.float32((d1_blue * (g1r2) + d2_blue * (g2r2))/((g1r2+g2r2)))  
    fused_base2 = np.float32((b1_green * g1r1 + b2_green * g2r1)/((g1r1+g2r1)))   
    fused_detail2 = np.float32((d1_green * (g1r2) + d2_green * (g2r2))/((g1r2+g2r2)))    
    fused_base3 = np.float32((b1_red * (g1r1) + b2_red * (g2r1))/((g1r1+g2r1)))
    fused_detail3 = np.float32((d1_red * (g1r2) + d2_red * (g2r2))/((g1r2+g2r2)))
        
        
    B1=np.float32(fused_base1+fused_detail1)
    B2=np.float32(fused_base2+fused_detail2)
    B3=np.float32(fused_base3+fused_detail3)
    
    fusion1=np.float32(cv2.merge((B1, B2, B3)))
    fusion1=fusion1/fusion1.max()
    return fusion1