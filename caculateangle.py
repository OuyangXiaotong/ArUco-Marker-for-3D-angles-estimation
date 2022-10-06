import math

import estimatepose
import numpy as np

I=[[1,0,0],
   [0.1,0],
   [0,0,1]]
#dist=estimatepose.dist
dist=[0.16285088, -0.22159294,  0.00103373,  0.0010233,   0.0743687]
#newcameramtx=estimatepose.newcameramtx
   # [[1.78165613e+03,0.00000000e+00,1.13784789e+03],
   #                       [0.00000000e+00,8.97238672e+03,3.19104003e+02],
   #                       [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
#mtx =estimatepose.mtx
   # [[375.59232499,0,326.65323449],
   #              [0,       374.89607017,183.20463121],
   #              [0,         0,          1]]
rvec=estimatepose.rvec[0][0]
kvalue=0
rvec=rvec[0][0]
for i in range(rvec.lenth):
   kvalue=kvalue+rvec[i]**2
kvalue=kvalue**0.5
kx=rvec[0]/kvalue
ky=rvec[1]/kvalue
kz=rvec[2]/kvalue
R3K=[[0,-kz,ky],
     [kz,0,kx],
     [-ky,kx,0]]
R1=kvalue*I
K=np.matrix([kx,ky,kz])
KT=K.T
R2=(1-math.cos(kvalue))*(KT*K)
R3=math.sin(kvalue)*R3K
R=R1+R2+R3


