

global NNDayDir, NNNightDir, dayModel, nightModel
NNDayDir, NNNightDir = '',''
dayModel, nightModel = None,None

def predict(images, df, Export, dayModelDir = '',nightModelDir = ''):
    import tensorflow as tf 
    from tensorflow import keras
    import cv2
    import numpy as np
    import os
    from datetime import datetime
    import time
    global NNDayDir, NNNightDir, dayModel, nightModel
    
    confs = []
    try:
        if NNDayDir != dayModelDir:
            NNDayDir = dayModelDir
            dayModel = keras.models.load_model(dayModelDir)
    except:
        p= 1
    try:
        if NNNightDir != nightModelDir:
            NNNightDir = nightModelDir
            nightModel = keras.models.load_model(nightModelDir)
    except:
        p = 1
    '''
    SABLE_108_July_2017=[155: 1655 757:2257]
    SABLE_108_May_2019=[27:1527 910:2410]
    SABLE_108_June_2019=[178:1678 1144:2644]
    SABLE_108_July_2019=[198:1698 783:2283]

    SABLE_405_July_2017=[640:2140 822:2322]
    SABLE_405_May_2019=[504:2004 822:2322]
    SABLE_405_June_2019=[341:1841 794:2294]
    SABLE_405_July_2019=[508:2008 755:2255]

    SEM_205_July_2017=[385:1885 676:2176]
    SEM_205_May_2019=[511:2011 773:2273]
    SEM_205_June_2019=[490:1990 760:2260]
    SEM_205_July_2019=[514:2014 835:2335]
    '''
    i = 0
    for image in images:   #For all images in directory
        img_array = image[1].giveCVImgarr()
        img_array = cv2.resize(img_array, (3264, 2448))
        

        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)  #Map to HSV

        mean = np.mean(hsv[:,:,0])    #Check Hue Channel to determine Night/Day

        if mean < 1:  #Nighttime
            crop_array = img_array[0:2250, 0:3264]
            rgb_array = crop_array[:,:,::-1]      #Process input for Night Model
            x = cv2.resize(rgb_array, (299, 299))
            x = x.astype('float32')
            x /= 255
            score =   nightModel.predict((x.reshape(1,299,299,3)))   #Predict and save
            prediction = np.argmax(score)
            
            now = datetime.now()
            current_time = now.isoformat()+' '+now.strftime("%H:%M:%S")

            image[1].predictions[NNNightDir.split('/')[-1]] = [score,prediction,current_time] 
            image[1].updatePredictions(NNNightDir.split('/')[-1])
            
            try:
                if df == None:
                    p = 1
            except:
                df.loc[i] = [image[1].imgpath.split('/')[-1], NNNightDir, prediction, np.max(score)]
                i += 1
            
        else:         #Daytime
            if image[0] == 'SABLE_108_July_2017':
                crop_array = img_array[155: 1655,757:2257]
            elif image[0] == 'SABLE_108_May_2019':
                crop_array = img_array[27:1527,910:2410]
            elif image[0] == 'SABLE_108_June_2019':
                crop_array = img_array[178:1678,1144:2644]
            elif image[0] == 'SABLE_108_July_2019':
                crop_array = img_array[198:1698,783:2283]

            elif image[0] == 'SABLE_405_July_2017':
                crop_array = img_array[640:2140,822:2322]
            elif image[0] == 'SABLE_405_May_2019':
                crop_array = img_array[504:2004,822:2322]
            elif image[0] == 'SABLE_405_June_2019':
                crop_array = img_array[341:1841,794:2294]
            elif image[0] == 'SABLE_405_July_2019':
                crop_array = img_array[508:2008,755:2255]
                
            elif image[0] == 'SEM_205_July_2017':
                crop_array = img_array[385:1885,676:2176]
            elif image[0] == 'SEM_205_May_2019':
                crop_array = img_array[511:2011,773:2273]
            elif image[0] == 'SEM_205_June_2019':
                crop_array = img_array[490:1990,760:2260]
            elif image[0] == 'SEM_205_July_2019':
                crop_array = img_array[514:2014,835:2335]
            rgb_array = crop_array[:,:,::-1]
            x = cv2.resize(rgb_array, (299, 299))
            x = x.astype('float32')
            x /= 255
            score =  dayModel.predict((x.reshape(1,299,299,3)))  #Predict and save
            prediction = np.argmax(score)

            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            
            image[1].predictions[NNDayDir.split('/')[-1]+"_"+str(current_time)] = [score,np.max(prediction),current_time] 
            image[1].updatePredictions(NNDayDir.split('/')[-1]+"_"+str(current_time))

            try:
                if df == None:
                    p = 1
            except:
                df.loc[i] = [image[1].imgpath.split('/')[-1], NNDayDir, prediction, np.max(score)]
                i += 1
        if Export:
            confs.append(score[0])
    return df,confs