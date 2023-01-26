from deepface import DeepFace
import cv2
class FaceRecog:
    def __init__(self,model="VGG-Face",metric="euclidean_l2",threshold = 0.86, detector_backend = "skip"): #"VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"
        self.model = model
        self.metric = metric
        self.threshold = threshold
        self.detector_backend = detector_backend
    def verify(self,frame,src):
        res = DeepFace.verify(frame,src,model_name=self.model,distance_metric = self.metric,detector_backend = self.detector_backend)
        if res["distance"]<=self.threshold:
            return (True,res["distance"])
        return (False,res["distance"])
        

# fr = FaceRecog()
# img1 = cv2.imread("/home/siddharth/Downloads/lbj1.jpeg")
# img2 = cv2.imread("/home/siddharth/Downloads/lbj2.jpeg")
# print(fr.verify(img1,img2))