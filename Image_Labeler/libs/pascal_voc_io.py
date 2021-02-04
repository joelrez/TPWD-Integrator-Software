#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from Image_Labeler.libs.constants import DEFAULT_ENCODING
from Image_Labeler.libs.ustr import ustr


XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING

class PascalVocWriter:

    def __init__(self, foldername, filename, time, date, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.time = time
        self.date = date
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.nnlist = []
        self.loclist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')
        else:
            top.set('verified', 'no')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        time = SubElement(top, 'time')
        time.text = self.time

        #date = SubElement(top, 'date')
        #date.text = self.date

        #source = SubElement(top, 'source')
        #database = SubElement(source, 'database')
        #database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        #depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        #if len(self.imgSize) == 3:
        #    depth.text = str(self.imgSize[2])
        #else:
        #    depth.text = '1'

        #segmented = SubElement(top, 'segmented')
        #segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['class'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def addPrediction(self,val,nnname):
        vals = {'prediction':val[1], 'confidence':val[0], 'datepredicted': val[2], 'name' : nnname}
        self.nnlist.append(vals)

    def addLoc(self,val,nnname):
        vals = {'BBOut':val[0], 'datepredicted': val[1], 'name' : nnname}
        self.loclist.append(vals)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'class')
            name.text = ustr(each_object['class'])
            #pose = SubElement(object_item, 'pose')
            #pose.text = "Unspecified"
            #truncated = SubElement(object_item, 'truncated')
            #if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
            #    truncated.text = "1" # max == height or min
            #elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
            #    truncated.text = "1" # max == width or min
            #else:
            #    truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
        
        for each_object in self.nnlist:
            #{'prediction':val[1], 'confidence':[val[0][0],val[0][1]], 'date predicted': val[2], 'name' : nnname}
            nnobj = SubElement(top, 'nn')
            name = SubElement(nnobj, 'name')
            name.text = str(each_object['name'])
            pred = SubElement(nnobj, 'prediction')
            pred.text = str(each_object['prediction'])
            conf = SubElement(nnobj, 'confidence')
            conf.text = str(each_object['confidence'])
            date = SubElement(nnobj, 'datepredicted')
            date.text = str(each_object['datepredicted'])

        for each_object in self.loclist:
            nnobj = SubElement(top, 'loc')
            name = SubElement(nnobj, 'name')
            name.text = str(each_object['name'])
            try:
                shape = each_object['BBOut'].shape
                for BBOut in each_object['BBOut']:
                    pred = SubElement(nnobj, 'BBOut')
                    xmin = SubElement(pred,'xmin')
                    xmin.text = str(BBOut[0])
                    ymin = SubElement(pred,'ymin')
                    ymin.text = str(BBOut[1])
                    xmax = SubElement(pred,'xmax')
                    xmax.text = str(BBOut[2])
                    ymax = SubElement(pred,'ymax')
                    ymax.text = str(BBOut[3])
            except:
                BBOut = each_object['BBOut']
                pred = SubElement(nnobj, 'BBOut')
                xmin = SubElement(pred,'xmin')
                xmin.text = str(BBOut[0])
                ymin = SubElement(pred,'ymin')
                ymin.text = str(BBOut[1])
                xmax = SubElement(pred,'xmax')
                xmax.text = str(BBOut[2])
                ymax = SubElement(pred,'ymax')
                ymax.text = str(BBOut[3])
            date = SubElement(nnobj, 'datepredicted')
            date.text = str(each_object['datepredicted'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.locs = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))
    
    def getLocs(self):
        return self.locs
   
    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
            elif verified == 'no':
                self.verified = False
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('class').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        for object_iter in xmltree.findall('loc'):
            for bndbox in object_iter.findall("BBOut"):
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                self.locs.append(('Bird', points, None, None, 0))
        return True
