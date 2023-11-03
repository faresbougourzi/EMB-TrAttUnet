# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:04:11 2023

@author: FaresBougourzi
"""
import cv2
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

kernel1 = np.ones((7,7),np.uint8)
criterion = CrossEntropyLoss()
TFtotensor = ToTensorV2()

def Hybrid_loss(GT_Inf, Pred_Inf):
    Ed = np.array(GT_Inf).copy()
    Ed = Ed.astype(np.uint8)
    Edge = cv2.morphologyEx(Ed, cv2.MORPH_GRADIENT, kernel1)
    Edge[Edge > 0.0] = 1.0
    
    lung_inf  = Ed.copy()
    lung_inf = np.where(lung_inf < 2, 0, lung_inf)
    lung_inf = np.where(lung_inf > 1, 1, lung_inf) 
    
    lung_inf2  = Ed.copy()
    lung_inf2 = np.where(lung_inf2 > 1, 0, lung_inf2)

    yc = lung_inf.astype(np.uint8)
    yg = lung_inf2.astype(np.uint8)
    
    kernel = np.ones((7, 7),np.uint8)
    yc2 = cv2.morphologyEx(yc, cv2.MORPH_GRADIENT, kernel)
    yg2 = cv2.morphologyEx(yg, cv2.MORPH_GRADIENT, kernel)
    
    Edge = yg2 +yc2
    Edge = np.where(Edge > 1, 1, Edge) 
    Edge = TFtotensor(Edge)
        
    loss1 = criterion(GT_Inf, Pred_Inf)
    loss2 = criterion(GT_Inf*Edge, Edge) 
    
    loss = loss1 + 2*loss2
    
    return loss

