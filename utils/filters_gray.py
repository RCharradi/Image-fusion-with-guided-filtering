# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:29:16 2018

@author: Ramzi Charradi
"""
from scipy.ndimage.filters import laplace
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Algorithm using the implemented guided Filter for gray_scale images

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
    eps_2=1e-6
    
   
    if im1.max()>1:
        im1=im1/255
    if im2.max()>1:
        im2=im2/255
        
    base_layer1 = uniform_filter(im1, mode='reflect',size=average_filter_size)
    base_layer2 = uniform_filter(im2, mode='reflect',size=average_filter_size)
    
  
    detail_layer1 = im1 - base_layer1
    detail_layer2 = im2 - base_layer2
        
    saliency1 = gaussian_filter(abs(laplace(im1,mode='reflect')), sigma_r,mode='reflect')
    saliency2 = gaussian_filter(abs(laplace(im2,mode='reflect')),sigma_r,mode='reflect')
    
    mask = np.argmax([saliency1, saliency2], axis=0)
    
    g1r1 = guided_filter(1 - mask , im1, r_1, eps_1)
    g2r1 = guided_filter(mask, im2, r_1, eps_1)
    g1r2 = guided_filter(1 - mask, im1, r_2, eps_2)
    g2r2 = guided_filter(mask, im2 , r_2, eps_2)
    
    fused_base = (base_layer1 * (g1r1) + base_layer2 * (g2r1))/(g1r1+g2r1) 
    fused_detail = (detail_layer1 * (g1r2) + detail_layer2 * (g2r2))/(g1r2+g2r2) 
    
    fusion = fused_base + fused_detail
        
    
    return fusion