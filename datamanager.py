import pandas
import numpy as np 
import cv2
import os
import shutil
from PIL import Image
import Day_Night_python as dnp
from pathlib import Path
import xml.etree.ElementTree as ET 
import Image_Labeler.libs.pascal_voc_io as pascal
from xml.dom import minidom
global df,rootfp

class img:
    def __init__(self,row):
        self.xml = row['XML']
        self.timegroup = row['YYYY/MM']
        self.DSName = row['DSName']
        img = Image.open(row['Image Path'])
        imgExif = img._getexif()
    
        self.datetime = imgExif[306].split(' ')
        self.datetime[0] = self.datetime[0].replace(':','-')
        self.datetime = self.datetime[0]+' '+self.datetime[1]
        self.imgpath = row['Image Path']
        self.predictions = {}
        try:
            xml = minidom.parse(self.xml)
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
                self.predictions[channel.childNodes[1].childNodes[0].nodeValue] = temp
        except:
            p = 1
        if row['Day/Night'] == 'Day':
            self.day = True
        else:
            self.day = False
    def humanLabeled(self):
        try:
            return str(len(minidom.parse(self.xml).getElementsByTagName('object')) > 0)
        except:
            return str(False)
    
    def nnLabeled(self):
        try:
            lenNNList = len(minidom.parse(self.xml).getElementsByTagName('nn'))
            lenLocList = len(minidom.parse(self.xml).getElementsByTagName('loc'))
            if lenNNList > 0 or lenLocList > 0:
                return str(True)
            else:
                return str(False)
        except:
            return str(False)

    def getLabel(self):
        try:   
            attr = ''
            attr += 'Image Name: '+self.imgpath.split('/')[-1]+'\n'
            attr += 'Date Taken: '+self.datetime+'\n'
            attr += 'Neural Network Predicted: '+self.nnLabeled()+'\n'
            attr += 'Human Labeled: '+self.humanLabeled()+'\n'
            dims = str(self.getDims())
            dims = dims[:dims.rfind(',')]
            dims += ')'
            attr += 'Dimensions: '+dims+'\n'
        finally:
            return attr

    
    def getDims(self):
        return np.asarray(Image.open(self.imgpath)).shape
    
    def showImg(self):
        imga = Image.open(self.imgpath)
        imga.show()

    def updatePredictions(self,network):
        #get shapes and readd them.
        shapes = []
        try:
            xml = minidom.parse(self.xml)
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

        elems = self.xml.split('/')
        imgFolderName = ''
        for elem in range(len(elems)-1):
            imgFolderName+= elems[elem]+'\\'
        writer = pascal.PascalVocWriter(imgFolderName, self.xml.split('.')[0], self.datetime, self.datetime.split(' ')[0],
                                 self.getDims())
        for shape in shapes:
            #xmin, ymin, xmax, ymax, name, difficult
            writer.addBndBox(shape['xmin'],shape['ymin'],shape['xmax'],shape['ymax'],shape['name'],shape['difficult'])

        for prediction in self.predictions:
            writer.addPrediction(self.predictions[prediction],prediction)
        writer.save()

    def giveCVImgarr(self):
        return cv2.imread(self.imgpath)

def createDSDir(filepath):
    os.makedirs(filepath+'/imgsfolder')
    os.makedirs(filepath+'/XMLs folder')
    df = pandas.DataFrame(columns=['Image Path','XML','YYYY/MM','DSName','Day/Night'])   #Results format
    df.to_csv(filepath+'/Main.csv', encoding='utf-8', index=False)
    return openDS(filepath+'/Main.csv')

def copyDSintoDS(src,fileTree):
    global rootfp
    for ds in os.listdir(src+'/imgsfolder'):
        fileTree = copyintoDS(src+'/imgsfolder/'+ds,fileTree)
        for xml in os.listdir(src+'/XMLs folder/'+ds):
            preds,nnames = getPredictions(src+'/XMLs folder/'+ds+'/'+xml)
            i = 0
            while i < len(preds):
                conf = preds[i][0]
                if '\t' in conf:
                    conf = np.array([conf.split('\t')[0][2:],conf.split('\t')[1][:-2]]).astype(float)
                else:    
                    conf = np.array([conf.split(' ')[0][2:],conf.split(' ')[1][:-2]]).astype(float)
                preds[i][0] = conf
                fileTree[rootfp+'imgsfolder/'+ds+'/'+xml.split('.')[0]+'.JPG'].predictions[nnames[i]] = preds[i]
                i += 1
            os.remove(rootfp+'XMLs folder/'+ds+'/'+xml)
            copyfile(src+'/XMLs folder/'+ds+'/'+xml, rootfp+'XMLs folder/'+ds+'/'+xml)
    return fileTree

def openDS(csvfilepath):
    #Open the given csv file.
    global df,rootfp
    df = pandas.read_csv(csvfilepath)
    fileTree = {}

    #Checks if the XMLs folder or imgsfolder exists in the DS folder. If not, create the directories and return None because there's obviously no
    #images in the DS. great.
    rootfp = ''
    temprootcomps = csvfilepath.split('/')
    for comp in range(len(temprootcomps)-1):
        rootfp += temprootcomps[comp]+'/'

    if not os.path.exists(rootfp+'imgsfolder') or not os.path.exists(rootfp+'XMLs folder'):
        os.makedirs(rootfp+'imgsfolder')
        os.makedirs(rootfp+'XMLs folder')
        return fileTree

    #Iterate through entries in the CSV
    for _,row in df.iterrows():
        #Add entry to fileTree
        fileTree[row['Image Path']] = img(row)
        if not Path(rootfp+'XMLs folder/'+row.DSName+'/'+row['Image Path'].split('.')[0].split('/')[-1]+'.xml').exists():
            f = open(rootfp+'XMLs folder/'+row.DSName+'/'+row['Image Path'].split('.')[0].split('/')[-1]+'.xml','x')
            f.close()

    #return fileTree.
    return fileTree

def getPredictions(xmlPath):
    try:
        predictions, nnnames = [],[]
        xml = minidom.parse(xmlPath)
        root = xml.getElementsByTagName('nn')
        for channel in root:
            temp = [0]*3

            nnnames.append(channel.childNodes[1].childNodes[0].nodeValue)
            temp[1] = channel.childNodes[3].childNodes[0].nodeValue
            temp[0] = channel.childNodes[5].childNodes[0].nodeValue
            temp[2] = channel.childNodes[7].childNodes[0].nodeValue
            
            predictions.append(temp)
    finally:
        return predictions, nnnames
'''
def copyloopercode():

def copyhelper(filepath):
'''

def scheme(filepath):
    img = Image.open(filepath)
    filepath2 = filepath.split('.')[-2].split('(')[0]+'.JPG'
    if filepath[0] == '.':
        filepath2 = '.'+filepath2
    newfilepath = filepath.split('/')[-2]+'_'

    #Go into image metadata to get the time taken.
    imgExif = img._getexif()
    datetime = imgExif[306].split(' ')
    time = datetime[1].split(':')
    img.close()
    
    newfilepath += datetime[0].split(':')[-1]+'_'+time[0]+'_'+time[1]+'_'+filepath.split('/')[-1]
    return newfilepath

def copyintoDS(filepath,fileTree):
    global df,rootfp
    i = len(df.loc[:])+1
    #Iterate through folders in the given filepath and make an entry
    imgs = os.listdir(filepath)
    dsName = filepath.split('/')[-1]
    try:
        os.makedirs(rootfp+'imgsfolder/'+dsName)
        os.makedirs(rootfp+'XMLs folder/'+dsName)
    except:
        p = 1
    for imgstr in imgs:
        if imgstr.split('.')[-1].upper() == 'JPG':
            newimgstr = scheme(filepath+'/'+imgstr)
            try:
                open(rootfp+'XMLs folder/'+dsName+'/'+newimgstr.split('.')[0]+'.xml', 'x')
            except:
                p = 1
            
            try:
                os.rename(filepath+'/'+imgstr,rootfp+'imgsfolder/'+dsName+'/'+newimgstr)
                
            except:
                p = 1
            DNIndic = "Night"
            if dnp.isDay(rootfp+'imgsfolder/'+dsName+'/'+newimgstr):
                DNIndic = "Day"

            imgx = Image.open(rootfp+'imgsfolder/'+dsName+'/'+newimgstr)
            imgExif = imgx._getexif()
            timegroup = imgExif[306]
            timegroup = timegroup.split(' ')[0].split(':')
            timegroup = timegroup[0]+'/'+timegroup[1]
            imgx.close()

            df.loc[i] = [rootfp+'imgsfolder/'+dsName+'/'+newimgstr, rootfp+'XMLs folder/'+dsName+'/'+newimgstr.split('.')[0]+'.xml','..',filepath.split('/')[-1],DNIndic]
            fileTree[rootfp+'imgsfolder/'+dsName+'/'+newimgstr] = img({'Image Path' : rootfp+'imgsfolder/'+dsName+'/'+newimgstr, 'XML': rootfp+'XMLs folder/'+dsName+'/'+newimgstr.split('.')[0]+'.xml', 'YYYY/MM' : '..', 'DSName' :filepath.split('/')[-1], 'Day/Night' : DNIndic})
            i += 1
        else:
            continue
        df.to_csv(rootfp + 'Main.csv', encoding='utf-8', index=False)
    return fileTree
