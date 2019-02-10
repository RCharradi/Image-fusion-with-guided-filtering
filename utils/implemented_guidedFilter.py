# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:29:16 2018

@author: Ramzi Charradi
"""

# Algorithm using the implemented guided Filter


from scipy.ndimage.filters import laplace
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')
    
def padding(im,r):
    return np.pad(im,((r, r),(r, r)), 'reflect')

def guided_filter(im,guide,r,epsilone):
    im = np.array(im, np.float32)
    a = np.zeros((im.shape[0],im.shape[1]), np.float32)
    b = np.zeros((im.shape[0],im.shape[1]), np.float32)
    O = np.array(im, np.float32, copy=True)
    im=padding(im,r)
    guide=padding(guide,r)   
    n=np.shape(im)[0]
    m=np.shape(im)[1] 
    a_k = np.zeros((n,m), np.float32)
    b_k = np.zeros((n,m), np.float32)
    w=2*r+1
    for i in range(r,n-r):
        for j in range(r,m-r):
            I=guide[i-r:i+r+1 ,j-r:j+r+1 ]
            P=im[i-r:i+r+1 ,j-r:j+r+1 ]
            mu_k = np.mean(I)
            delta_k = np.var(I)
            P_k_bar = np.mean(P)
            somme = np.dot(np.ndarray.flatten(I), np.ndarray.flatten(P))/(w**2)
            a_k[i,j] = (somme - mu_k * P_k_bar) / (delta_k + epsilone)
            b_k[i,j] = P_k_bar - a_k[i,j] * mu_k   
    a=a_k[r:n-r+1,r:m-r+1]
    b=b_k[r:n-r+1,r:m-r+1]
    a=padding(a,r)
    b=padding(b,r)
    for i in range(r, n-r):
        for j in range(r, m-r):
            a_k_bar = a[i-r : i+r+1, j-r : j+r+1].sum()/(w*w)
            b_k_bar = b[i-r : i+r+1, j-r : j+r+1].sum()/(w*w)
            O[i-r,j-r] = a_k_bar * guide[i,j] + b_k_bar
    return O
                
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
    
    
    saliency1 = gaussian_filter(abs(laplace(im1_blue+im1_green+im1_red,mode='reflect')),sigma_r,mode='reflect')
    saliency2 = gaussian_filter(abs(laplace(im2_blue+im2_green+im2_red,mode='reflect')),sigma_r,mode='reflect')
    mask = np.float32(np.argmax([saliency1, saliency2], axis=0))
    
    im1=np.float32(im1)
    im2=np.float32(im2)
    
    g1r1 = guided_filter(1 - mask, im1[:,:,0], r_1, eps_1)+guided_filter(1 - mask, im1[:,:,1], r_1, eps_1)+guided_filter(1 - mask, im1[:,:,2], r_1, eps_1)
    
    g2r1 = guided_filter(mask, im2[:,:,0], r_1, eps_1)+guided_filter(mask, im2[:,:,1], r_1, eps_1)+guided_filter(mask, im2[:,:,2], r_1, eps_1)
    g1r2 = guided_filter(1 - mask, im1[:,:,0], r_2, eps_2)+ guided_filter(1 - mask, im1[:,:,1], r_2, eps_2)+ guided_filter(1 - mask, im1[:,:,2], r_2, eps_2)
    g2r2 =  guided_filter(mask, im2[:,:,0], r_2, eps_2)+ guided_filter(1 - mask, im2[:,:,1], r_2, eps_2)+ guided_filter(1 - mask, im2[:,:,2], r_2, eps_2)

    
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