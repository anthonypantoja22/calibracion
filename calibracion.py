# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
 
# termination criterio
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
 
# preparar puntos de objeto, como (0,0,0,0), (1,0,0,0), (2,0,0,0)...., (6,5,0)
objp = np.zeros((8*8,3), np.float32)
objp[:,:2] = 19*np.mgrid[0:8,0:8].T.reshape(-1,2)
# print(objp)
# Arrays para almacenar puntos de objeto y puntos de imagen de todas las imágenes.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane
images = glob.glob('calibracion/prueba2/*.jpg')

i = 1 
for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  h,  w = img.shape[:2]

  # Encuentra las esquinas del tablero de ajedrez
  ret, corners = cv2.findChessboardCorners(gray, (8,8),None)
  
  # Si se encuentran, añada puntos de objeto, puntos de imagen (después de refinarlos)
  if ret == True:
    objpoints.append(objp)
 
    corners2 = cv2.cornerSubPix(gray,corners,(17,17),(-1,-1),criteria)
    imgpoints.append(corners2)  
 
    # Dibuja y muestra las esquinas
    img = cv2.drawChessboardCorners(img, (8,8), corners2,ret)
    #cv2.imshow('img',img)
    #cv2.imwrite('resultado.jpg',img)
    #cv2.waitKey(0)
    # print("obj=",objpoints)

    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# print("ret = ", ret)
# print("")
# print("mtx = " , mtx)
# print("")
# print("dist =" , dist)
# print("")
# print("rvecs = ", rvecs)
# print("")
# print("tvecs = ", tvecs )
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
mean_error = 0 
for i in range(len(objpoints)): 
  imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) 
  error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
  mean_error= mean_error+error 
print ("total error: ", mean_error/len(objpoints)) 



for rname in images:
    img = cv2.imread(rname)
    print("%s"%rname)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    cv2.imshow('img',dst)
    cv2.imwrite('resultado de calibracion/resultado de calibracion 1/%s'%rname,dst)
    cv2.waitKey(0)
cv2.destroyAllWindows()