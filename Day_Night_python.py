import numpy as np
import cv2

def isDay(filename):
  img_array = cv2.imread(filename)

  crop_array = img_array[0:2250, 0:3264]   #Crop away infobar

  hsv = cv2.cvtColor(crop_array, cv2.COLOR_BGR2HSV)  #Map to HSV

  mean = np.mean(hsv[:,:,0])    #Check Hue Channel to determine Night/Day
  
  if  mean < 1:
    return False
  else:
    return True
                
                
        
        
        

            