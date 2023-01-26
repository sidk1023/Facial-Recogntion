import os
import cv2
import numpy as np
from frameDiff import FrameDiff
from faceDetect import FaceDetect
from faceRecog import FaceRecog
from antiSpoof import Antispoof

img1 = cv2.resize(cv2.imread("/home/siddharth/Downloads/lbj1.jpeg"),(300,300))
img2 = cv2.resize(cv2.imread("/home/siddharth/Downloads/lbj2.jpeg"),(300,300))
img3 = cv2.resize(cv2.imread("/home/siddharth/Downloads/face.jpeg"),(300,300))
diff = FrameDiff(0.9)
detect = FaceDetect(0.50,"cpu")
rec = FaceRecog()

print("diff",diff.ssim(img1,img2))
print("detect",detect.detect(img3))
print("rec",rec.verify(img1,img2))