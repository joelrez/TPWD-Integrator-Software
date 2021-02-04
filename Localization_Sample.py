#!/usr/bin/env python
"""
Sample script that uses the BirdLocalizationApp module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

from __future__ import print_function
import BirdLocalizationApp
import matlab
from datetime import datetime
import numpy as np
import pandas 

def updateLocalizations(img_pathIn, xmlpath, val, model_pathIn):
    from xml.dom import minidom
    import Image_Labeler.libs.pascal_voc_io as pascal
    import numpy as np
    from PIL import Image

    img = Image.open(img_pathIn)
    imgExif = img._getexif()
    datetime = imgExif[306].split(' ')
    imgarray = np.asarray(img)
    img.close()

    writer = pascal.PascalVocWriter('', xmlpath.split('.')[0], datetime[0]+' '+datetime[1], datetime[0],
        imgarray.shape)

    shapes = []
    try:
        xml = minidom.parse(xmlpath)
        channels = xml.getElementsByTagName('object')
        for channel in channels:
            temp = {}
            #xmin, ymin, xmax, ymax, name, difficult
            temp['name'] = channel.getElementsByTagName('class')[0].childNodes[0].nodeValue
            temp['xmin'] = channel.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
            temp['ymin'] = channel.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
            temp['xmax'] = channel.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
            temp['ymax'] = channel.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
            temp['difficult'] = channel.getElementsByTagName('difficult')[0].childNodes[0].nodeValue
            shapes.append(temp)
    except:
        p = 1

    predictions = {}
    try:
        xml = minidom.parse(xmlpath)
        root = xml.getElementsByTagName('nn')

        for channel in root:
            temp = [0]*3
            temp[1] = channel.childNodes[3].childNodes[0].nodeValue
            conf = channel.childNodes[5].childNodes[0].nodeValue
            if '\t' in conf:
                conf = np.array([conf.split('\t')[0][2:],conf.split('\t')[1][:-2]]).astype(float)
            else:    
                conf = np.array([conf.split(' ')[0][2:],conf.split(' ')[1][:-2]]).astype(float)
            temp[0] = conf
            temp[2] = channel.childNodes[7].childNodes[0].nodeValue
            predictions[channel.childNodes[1].childNodes[0].nodeValue] = temp
    except:
        p = 1

    try:
        xml = minidom.parse(xmlpath)
        root = xml.getElementsByTagName('loc')

        for channel in root:
            tempVal = []
            locName = channel.childNodes[1].childNodes[0].data
            for BBOut in channel.getElementsByTagName('BBOut'):
                tempVal.append([float(BBOut.childNodes[1].childNodes[0].data),
                                float(BBOut.childNodes[3].childNodes[0].data),
                                float(BBOut.childNodes[5].childNodes[0].data),
                                float(BBOut.childNodes[7].childNodes[0].data)])
            tempVal.append(locName)
            locs[locName] = np.asarray(tempVal)
    except:
        p = 1
    
    locs = {model_pathIn.split('\\')[-1].split('.')[0]:val}
    
    for shape in shapes:
        #xmin, ymin, xmax, ymax, name, difficult
        writer.addBndBox(shape['xmin'],shape['ymin'],shape['xmax'],shape['ymax'],shape['name'],shape['difficult'])

    for prediction in predictions:
        writer.addPrediction(predictions[prediction], prediction)

    for loc in locs:
        writer.addLoc(locs[loc], loc)

    writer.save()

def localize(img_pathsIn,model_pathIn):
    

    df = pandas.DataFrame(columns=['Image Path','BBOut'])
    my_BirdLocalizationApp = BirdLocalizationApp.initialize()
    j = 0
    for img_pathIn in img_pathsIn:
        BBOut = my_BirdLocalizationApp.LocalizationModel([img_pathIn], [img_pathIn.split("/")[-2]], model_pathIn)
        BBOut = np.asarray(BBOut)
        df.loc[j] = [img_pathIn, BBOut]
        BBOutAdj = np.zeros((BBOut.shape[1],4))
        for i in range(BBOut.shape[1]):
            BBOutAdj[i,:2] = BBOut[0,i,:2]
            BBOutAdj[i,2:] = BBOut[0,i,:2] + BBOut[0,i,2:]
        df.to_csv('Results.csv', encoding='utf-8', index=False)
        j += 1  
        xmlpath = findXMLPath(img_pathIn)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        updateLocalizations(img_pathIn, xmlpath, [BBOutAdj,current_time], model_pathIn)
        j+=1
    
    my_BirdLocalizationApp.terminate()

def findXMLPath(img_pathIn):
    import os

    temp = img_pathIn.split('/')[:-3]
    xmlpath = temp[0]+'\\'
    temp.pop(0)
    for elemTemp in temp:
        xmlpath = os.path.join(xmlpath,elemTemp)
    temp = img_pathIn.split('/')[-3:]
    xmlpath = os.path.join(xmlpath, 'XMLs folder', temp[-2], temp[-1].split('.')[0]+'.xml')
    
    return xmlpath

'''paramFile = open('parameters.txt', 'r')
lines = paramFile.readlines()
paramFile.close()

img_pathsIn = lines[0][:-1].replace('/','\\').replace(' ','').replace('\'','').split(',')
folders = []
for img_pathIn in img_pathsIn:
    folders.append(img_pathIn.split('\\')[-2])
model_pathIn = lines[1]

localize(img_pathsIn,folders, model_pathIn)'''