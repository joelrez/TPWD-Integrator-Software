try:
    import Tkinter 
    import ttk
except ImportError: 
    import tkinter as Tkinter
    import tkinter.ttk as ttk
from tkinter import filedialog
import os
import datetime
from pathlib import Path
import datamanager as dm
import numpy as np
import Image_Labeler.labelImg as labelImg
import sys
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
from Image_Labeler.libs.ustr import ustr
from PIL import Image,ImageTk
import readonly
import pandas
import time
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from shutil import copyfile
import NNModel as NNM

#Current directory of this program
global curdir,gui,appname
curdir = os.path.dirname(os.path.realpath(__file__))
#Name of the application
appname = 'Dataset Viewer'
selLocal = False
modeChosen = False

#Class that pops up a table of predictions made corresponding to the provided image name
class tablePopup:

    def __init__(self,curitemName):
        global gui
        self.curitemName = curitemName
        self.entry = gui.fileTree[curitemName]
    
    def display(self):
        popup = Tkinter.Tk()
        popup.title(self.curitemName)
        popup.wm_geometry("250x250")
        
        #Display the treeview
        self.tablePopupTreeView = ttk.Treeview(popup, columns=('Neural Network', 'Date','Label'))
        self.tablePopupTreeView.heading('#0', text='Neural Network')
        self.tablePopupTreeView.heading('#1', text= 'Date')
        self.tablePopupTreeView.heading('#2', text = 'Label')
        self.tablePopupTreeView.heading('#3', text = 'Confidence')
        self.tablePopupTreeView.column('#0', stretch=Tkinter.YES)
        self.tablePopupTreeView.column('#1', stretch=Tkinter.YES)
        self.tablePopupTreeView.column('#2', stretch=Tkinter.YES)
        self.tablePopupTreeView.column('#3', stretch=Tkinter.YES)
        self.tablePopupTreeView.grid(row= 0, column = 1,sticky = 'nsew')

        #Show the predictions
        for prediction in self.entry.predictions:
            modelName = "_".join(prediction.split("_")[0:-1])
            vals = self.entry.predictions[prediction]
            score = vals[0]
            predictionval = vals[1]
            datetimeval = vals[2]
            values = ()
            if predictionval == '0' or predictionval == 0:
                values=(datetimeval,'No Animal(0)',np.max(score))
            else:
                values=(datetimeval,'Animal(1)',np.max(score))
            self.tablePopupTreeView.insert('', 'end', text = modelName,values = values)
        popup.mainloop()

class DSPopup:
    def __init__(self, curitemName):
        global gui
        popup = Tkinter.Toplevel()
        popup.wm_title(curitemName)
        popup.wm_geometry("250x250")
        popup.resizable(False, False)
        attstr = gui.fileTree[curitemName].getLabel()   
        label = Tkinter.Label(popup, text = attstr, justify = 'left')
        label.grid(row = 1, column = 0)
        tb = tablePopup(curitemName)
        butt1 = ttk.Button(popup, text="Predictions", command=tb.display)
        butt1.grid(row=5, column=0)
        butt = ttk.Button(popup, text="Show Image", command=gui.fileTree[curitemName].showImg)
        butt.grid(row=10, column=0)

class initialPopup:
    def __init__(self, curitemName):
        global gui,selLocal
        self.blue = '#776BFF'
        self.popup = Tkinter.Toplevel()
        self.popup.wm_title(curitemName)
        self.popup.wm_geometry("222x50")
        self.popup.resizable(False, False)
        self.butt1 = Tkinter.Button(self.popup, text="Predict", command=self.selPredict, background = self.blue)
        self.butt1.grid(row=5, column=0)
        
        self.butt2 = Tkinter.Button(self.popup, text="Localize", command=self.selLocaliz)
        self.butt2.grid(row=5, column=1)
        self.butt3 = Tkinter.Button(self.popup, text="OK", command=self.start)
        self.butt3.grid(row=5, column=2)

    def selPredict(self):
        global selLocal
        if selLocal:
            orig_color = self.butt1.cget("bg")
            
            #Set mode
            selLocal = False
            
            #Set color
            self.butt1.configure(background = self.blue)
            self.butt2.configure(background = orig_color)
    
    def selLocaliz(self):
        global selLocal
        if not selLocal:
            orig_color = self.butt2.cget("bg")

            #Set mode
            selLocal = True
            
            #Set color
            self.butt1.configure(background = orig_color)
            self.butt2.configure(background = self.blue)

    def start(self):
        global modeChosen,gui
        modeChosen = True
        gui.nnbuttons()
        self.popup.destroy()

class DatasetView(Tkinter.Frame):

    #Initializes the GUI
    def __init__(self, parent):
        global curdir
        Tkinter.Frame.__init__(self, parent)
        
        #Neural Network(NN) Panel Variables
        
        #Used for lookup of ID and Group correspondence.
        self.NNID2imgs = {}
        self.NNimgs2ID = {} 

        self.selectedNN = [] #For multiselection feature
        self.NNPaths = [] #Contains the paths for NN from the model folder inside the directory of this program
        self.NNfileroot = curdir+'/models' #Holds the root directory of the NN models

        #Dataset(DS) Panel Variables

        #Used for lookup of ID and Group correspondence.
        self.DSimgs2ID = {} 
        self.DSID2imgs = {}

        self.selectedDS = []
        self.fileTree = {}
        self.currentDS = ''
        
        #Create the GUI
        self.parent=parent
        self.initialize_user_interface()
        
    def initialize_user_interface(self):
        #Set up window
        global appname
        self.localimported = False
        self.tfimported = False
        self.parent.title(appname)
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)


        #Puts menu options at the top of the window
        menubar = Tkinter.Menu(self.master)
        self.master.config(menu=menubar)
        
        #Creates the DSmenu cascade menu
        #Or drop down menu at the top of the software
        DSmenu = Tkinter.Menu(menubar)
        #Adds Commands to the Dataset Cascade menu
        DSmenu.add_command(label="Open Dataset Directory", command = self.InquireDSFile)
        DSmenu.add_command(label="Create Dataset Directory", command = self.makeDSDir)
        DSmenu.add_separator() #Seperator in the commands that can be seen in the cascade menu
        DSmenu.add_command(label="Dump Images into Current Dataset Directory", command = self.dirCopy2DS)
        DSmenu.add_command(label="Copy Dataset Directory into Current Dataset Directory", command = self.CopyDSintoCurrDS)
        DSmenu.add_separator()
        DSmenu.add_command(label="Remove Current Dataset Directory", command = self.deleteDatatreeview)
        DSmenu.add_command(label="Credits", command = self.showCredits) 
        menubar.add_cascade(label="Dataset", menu=DSmenu) #Adds cascade DSmenu to the menubar

        #Adds Datatreeview to the left side of the screen
        LeftPanel = Tkinter.PanedWindow() #Window that Datatreeview gets placed into
        LeftPanel.grid(sticky='nsew')
        self.Datatreeview = ttk.Treeview(LeftPanel, columns=('Date','',''),selectmode = "extended") #Initializes a treeview object
        self.Datatreeview.heading('#0', text='Item') #Adds a heading or column named "Item"
        self.Datatreeview.heading('#1', text = 'Date And Time Taken')
        self.Datatreeview.heading('#2', text = 'Network Labeled')
        self.Datatreeview.heading('#3', text = 'Human Labeled')
        self.Datatreeview.column('#2', stretch=Tkinter.YES)
        self.Datatreeview.column('#1', stretch=Tkinter.YES)
        self.Datatreeview.column('#0', stretch=Tkinter.YES)
        self.Datatreeview.column('#3', stretch=Tkinter.YES)
        #Maps an event to a function
        self.Datatreeview.bind('<Double-Button-1>',self.openProperties) #Double left click
        #self.Datatreeview.bind('<Return>', self.filterNNMods) #Press enter
        self.Datatreeview.bind('<ButtonRelease-1>',self.selectDS) #Single left click
        #Adds Datatreeview to the LeftPanel
        LeftPanel.add(self.Datatreeview)

        #Adds the NN panel to the right side of the window
        RightPanel = Tkinter.PanedWindow()
        RightPanel.grid(row = 0,column = 1, sticky = 'nsew')
        self.NNtreeview = ttk.Treeview(RightPanel, columns=('Type', 'Date'),selectmode = "extended")
        self.NNtreeview.heading('#0', text='Neural Network Name')
        self.NNtreeview.heading('#1', text= 'Last Trained')
        self.NNtreeview.heading('#2', text = 'Path')
        self.NNtreeview.column('#1', stretch=Tkinter.YES)
        self.NNtreeview.column('#0', stretch=Tkinter.YES)
        self.NNtreeview.column('#2', stretch=Tkinter.YES)
        self.NNtreeview.bind('<ButtonRelease-1>',self.selectNN) #Single left click
        RightPanel.add(self.NNtreeview)

        #Gets NN models and places them into the NNtreeview
        self.getNNDirs()

        ip = initialPopup("")
    
    def nnbuttons(self):
        #Adding prediction button below the right panel.
        RightPanel2 = Tkinter.PanedWindow()
        RightPanel2.grid(row = 0,column = 2, sticky = 'nsew')
        if not selLocal:
            butt = ttk.Button(RightPanel2, text="Predict", command= self.popupSeq)
            RightPanel2.add(butt)
            butt.grid(row = 0,column = 0)
        else:
            butt2 = ttk.Button(RightPanel2, text="Localize", command= self.localize)
            RightPanel2.add(butt2)
            butt2.grid(row = 1, column = 0)
            
            
        butt1 = ttk.Button(RightPanel2, text="Label", command= self.LabelImgs)
        RightPanel2.add(butt1)
        butt3 = ttk.Button(RightPanel2, text="View Localizations", command= self.localizationpopup)
        RightPanel2.add(butt3)
        butt3.grid(row = 1, column = 1)

        #Add logo
        canvas = Tkinter.Canvas(RightPanel2, width = 150, height = 150)   
        img = Tkinter.PhotoImage(file=curdir+"/icon.png")      
        canvas.create_image(80,80, image=img)
        canvas.image = img    
        
        #Adjusts the positions of the buttons and canvas.
        RightPanel2.add(canvas)
        butt1.grid(row = 0, column = 1)
        canvas.place(relx = 0,rely = .7, relwidth = 1, relheight = 1)
    
    def localize(self):
        import Localization_Sample as local
        CurNNs = self.NNtreeview.selection()
        CurImgs = self.Datatreeview.selection()
        if len(CurNNs) > 0 and len(CurImgs) > 0:
            locModels = []
            img_pathsIn = []

            for CurNN in CurNNs:
                if self.NNID2imgs[self.NNtreeview.parent(CurNN)].split('\\')[-1] == 'Localization':
                    locModels.append(CurNN)

            for CurImg in CurImgs:
                    img_pathsIn.append(self.DSID2imgs[CurImg])
            for locModel in locModels:
                model_pathIn = os.path.join(self.NNfileroot, 'Localization',self.NNID2imgs[locModel])
                local.localize(img_pathsIn, model_pathIn)
            sys.modules.pop('Localization_Sample')
            self.setupDatatreeview('')

    def selectDS(self,e):
        item = self.Datatreeview.focus()
        itemchildren = self.Datatreeview.get_children(item)
        if len(itemchildren) > 0:
            for itemchild in itemchildren:
                itemchildren2 = self.Datatreeview.get_children(itemchild)
                if len(itemchildren2) > 0:
                    for itemchild2 in itemchildren2:
                        itemchildren3 = self.Datatreeview.get_children(itemchild2)
                        if len(itemchildren3) > 0:
                            for itemchild3 in itemchildren3:
                                if not itemchild3 in self.selectedDS and not self.Datatreeview.item(itemchild3)['values'] == '': 
                                    self.selectedDS.append(itemchild3)
                                else:
                                    if itemchild3 in self.selectedDS:
                                        self.selectedDS.remove(itemchild3)
                        else:
                            if not itemchild2 in self.selectedDS and not self.Datatreeview.item(itemchild2)['values'] == '': 
                                self.selectedDS.append(itemchild2)
                            else:
                                if itemchild2 in self.selectedDS:
                                    self.selectedDS.remove(itemchild2)
                else:
                    if not itemchild in self.selectedDS and not self.Datatreeview.item(itemchild)['values'] == '': 
                            self.selectedDS.append(itemchild)
                    else:
                        if itemchild in self.selectedDS:
                            self.selectedDS.remove(itemchild)
            if item in self.selectedDS:
                self.selectedDS.remove(item)
            self.Datatreeview.selection_remove(item)
        else:
            if not item in self.selectedDS and not self.Datatreeview.item(item)['values'] == '':
                self.selectedDS.append(item)
            else:
                if item in self.selectedDS:
                    self.selectedDS.remove(item)
        self.Datatreeview.selection_set(self.selectedDS)

    def selectNN(self,e):
        item = self.NNtreeview.focus()
        if not item in self.selectedNN:
            self.selectedNN.append(item)
        else:
            self.selectedNN.remove(item)
        self.NNtreeview.selection_set(self.selectedNN)

    def showCredits(self):
        #Creates a popup with the following properties.
        popup = Tkinter.Toplevel()
        popup.wm_title('Credits')
        popup.wm_geometry("1000x50")
        popup.resizable(False, False)
        label = Tkinter.Label(popup, text = 'The labeler software is called LabelImg created by Tzutalin et al on Github. You can see the other contributors: https://github.com/tzutalin/labelImg/graphs/contributors', justify = 'left')
        label.grid(row = 1, column = 0)
        popup.attributes("-topmost", True)
    
    #Left in case we want to go back to filtering the NN models based on whether they are day or night images.
    def filterNNMods(self,h):
        #Get the items currently selected in the Datatreeview
        curComps = self.Datatreeview.selection()

        #Initialize the NNnumRequired as 0 and Dayloaded and Nightloaded as False.
        Dayloaded,Nightloaded = False,False
        self.NNnumRequired = 0
        
        #Remove entries from treeview
        self.NNID2imgs = {}
        self.NNimgs2ID = {}
        for entry in self.NNtreeview.get_children():
            self.NNtreeview.delete(entry)

        #Go through selected images.
        for curComp in curComps:
            #If a day image
            if self.DSID2imgs[self.Datatreeview.parent(curComp)].split('/')[-1] == 'Day' and not Dayloaded:
                self.loadNNClassModel('Daytime Models')
                self.NNnumRequired += 1
                Dayloaded = True
            elif self.DSID2imgs[self.Datatreeview.parent(curComp)].split('/')[-1] == 'Night' and not Nightloaded:
               self.loadNNClassModel('Nighttime Models')
               Nightloaded = True
               self.NNnumRequired += 1
            if Dayloaded and Nightloaded:
                break
            
    def loadNNClassModel(self, modelType):
        for dirp in self.NNPaths:
            dir1 = dirp[0]
            dirs2 = dirp[1:]
            if dir1.split('\\')[-1][0] != '.' and dir1.split('\\')[-1] == modelType:
                id = self.NNtreeview.insert('','end',text = dir1.split('\\')[-1])
                self.NNID2imgs[id] = dir1
                self.NNimgs2ID[dir1] = id
                for dir2 in dirs2:
                    if dir2.split('\\')[-1].split('.')[-1] == 'model':
                        info = Path(dir2).stat()
                        d = datetime.datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        idimp = self.NNtreeview.insert(id,'end',text = dir2.split('\\')[-1], values = (d,dir2))
                        self.NNimgs2ID[dir2.split('\\')[-1]] = idimp
                        self.NNID2imgs[idimp] = dir2.split('\\')[-1]
    
    #May need to be renamed
    def popupSeq(self):
        try:
            dm.rootfp
            global settings, returnSettings, popup, label
            settings = []
            returnSettings = {}
            blue = '#776BFF'

            def updateLabel():
                global label
                if len(settings) == 0:
                    label['text'] = 'Nothing'
                else:
                    label['text'] = ''
                    for i in range(len(settings)-1):
                        label['text'] += settings[i] + ','
                    label['text'] += settings[len(settings)-1]
                
            def setExport():
                global settings
                if not ('Export' in settings):
                    orig_color = Yesbutt1.cget('bg')
                    Yesbutt1.configure(background = blue)
                    Nobutt1.configure(background = orig_color)
                    if not 'Export' in settings:
                        settings.append('Export')
                        updateLabel()

            def unsetExport():
                global settings
                if 'Export' in settings: 
                    orig_color = Nobutt1.cget("background")
                    Nobutt1.configure(background = blue)
                    Yesbutt1.configure(background = orig_color)
                    if 'Export' in settings:
                        settings.remove('Export')
                        updateLabel()

            def setResults():
                global settings
                if not ('Results' in settings):
                    orig_color = Yesbutt2.cget("background")
                    Yesbutt2.configure(background = blue)
                    Nobutt2.configure(background = orig_color)
                    if not 'Results' in settings:
                        settings.append('Results')
                        updateLabel()

            def unsetResults():
                global settings
                if 'Results' in settings:
                    orig_color = Nobutt2.cget("background")
                    Nobutt2.configure(background = blue)
                    Yesbutt2.configure(background = orig_color)
                    if  'Results' in settings:
                        settings.remove('Results')
                        updateLabel()

            def setOkay():
                global settings, returnSettings, popup
                if len(settings) != 0:
                    def startPrediction():
                        if 'Export' in settings:
                            returnSettings['Export'] = exportDir
                        if 'Results' in settings:
                            returnSettings['Results'] = resultsDir + '/' +resultsEntryBox.get()+'.csv'
                        popup.destroy()
                        self.predict(returnSettings = returnSettings)

                    popup.destroy()
                    popup = Tkinter.Toplevel()
                    popup.wm_title('Pre-Prediction Settings')
                    popup.wm_geometry("500x250")
                    popup.attributes("-topmost", True)
                    
                    okbutt = ttk.Button(popup, text="Okay", command=startPrediction)
                    okbutt.grid(row=4, column=0)

                    if 'Export' in settings:
                        global exportLabel, exportDir
                        def changeExportDir():
                            global exportDir, exportLabel
                            exportDir = filedialog.askdirectory()
                            exportLabel['text'] = 'Export directory: '+exportDir

                        exportLabel = Tkinter.Label(popup, text = 'Export directory:', justify = 'left')
                        exportLabel.grid(row = 0, column = 0)

                        dirExportbutt = ttk.Button(popup, text="", command=changeExportDir)
                        dirExportbutt.grid(row=0, column=1)

                    if 'Results' in settings:
                        global resultsDir, resultsLabel

                        def changeResultsDir():
                            global resultsDir, resultsLabel
                            resultsDir = filedialog.askdirectory()
                            resultsLabel['text'] = 'Results directory: '+resultsDir

                        resultsLabel = Tkinter.Label(popup, text = 'Results directory:', justify = 'left')
                        resultsLabel.grid(row = 2, column = 0)

                        dirResultsbutt = ttk.Button(popup, text="", command=changeResultsDir)
                        dirResultsbutt.grid(row=2, column=1)

                        label = Tkinter.Label(popup, text = 'Results file name: ', justify = 'left')
                        label.grid(row = 3,column = 0)
                        resultsEntryBox = Tkinter.Entry(popup)
                        resultsEntryBox.grid(row = 3, column = 1)
                        label = Tkinter.Label(popup, text = '.csv', justify = 'left')
                        label.grid(row = 3,column = 2)
                else:
                    popup.destroy()
                    self.predict()

            def cancel():
                popup.destroy()
            #fix condition required.
            if len(self.NNtreeview.selection()) != 0:
                popup = Tkinter.Toplevel()
                popup.wm_title('Pre-Prediction Settings')
                popup.wm_geometry("500x250")
                popup.resizable(False, False)

                
                label = Tkinter.Label(popup, text = 'Would you like to export the images into organized subfolders?', justify = 'left')
                label.grid(row = 0, column = 0)
                Yesbutt1 = Tkinter.Button(popup, text="Yes", command=setExport)
                Yesbutt1.grid(row=1, column=0)
                Nobutt1 = Tkinter.Button(popup, text="No", command=unsetExport, bg = blue)
                Nobutt1.grid(row=1, column=1)

                label = Tkinter.Label(popup, text = 'Would you like to save the prediction results into a directory?', justify = 'left')
                label.grid(row = 2, column = 0)
                Yesbutt2 = Tkinter.Button(popup, text="Yes", command=setResults)
                Yesbutt2.grid(row=3, column=0)
                Nobutt2 = Tkinter.Button(popup, text="No", command=unsetResults, bg = blue)
                Nobutt2.grid(row=3, column=1)

                cancelbutt = ttk.Button(popup, text="Cancel", command=cancel)
                cancelbutt.grid(row=4, column=0)

                label = Tkinter.Label(popup, text = 'Nothing', justify = 'left')
                label.grid(row = 4, column = 1)
                okbutt = ttk.Button(popup, text="Okay", command=setOkay)
                okbutt.grid(row=4, column=2)
                popup.attributes("-topmost", True) #Places the popup in the foreground
        except:
            p = 1
   
    def predict(self,returnSettings = None):
        
            #Get selected from the two treeviews
            curNNs = self.NNtreeview.selection()
            curComps = self.Datatreeview.selection()
            gp = self.Datatreeview.parent

            try:
                if 'Results' in returnSettings:
                    xceldir = returnSettings['Results']
                    df = pandas.DataFrame(columns=['Item','Model','Label','Confidence'])   #Results format
                else:
                    df = None
            except:
                df = None

            try:
                if 'Export' in returnSettings:
                    Export = True
                else:
                    Export = False
            except:
                Export = False

            #Set bools to False(default)
            nightRequired,dayRequired = False,False
            
            images = []
            global fileloc
            fileloc = []
            tempNNs = {'Day' : None, 'Night' : None}

            NNnumRequired = 0
            
            #Show popup to indicate the NNs are currently predicting on the images that were selected.
            popup = Tkinter.Toplevel()
            popup.wm_title('NN Predicting')
            popup.wm_geometry("500x50")
            popup.resizable(False, False)
            label = Tkinter.Label(popup, text = 'The Neural Network(s) is/are predicting on selected components from the dataset.', justify = 'left')
            label.grid(row = 1, column = 0)
            popup.attributes("-topmost", True) #Places the popup in the foreground
            self.show() #Show updates. If this statement is not executed, the popup will show up at the end of the function.
            
            #Sees if there are both Day and Night images.
            for curComp in curComps:
                if self.fileTree[self.DSID2imgs[curComp]].day and not dayRequired:
                    dayRequired = True
                    NNnumRequired += 1
                elif not self.fileTree[self.DSID2imgs[curComp]].day and not nightRequired:
                    nightRequired = True
                    NNnumRequired += 1
            
            #Gets NNs into the tempNNs dictionary
            for curNN in curNNs:
                if self.NNID2imgs[self.NNtreeview.parent(curNN)].split('\\')[-1] == 'Daytime Models':
                    tempNNs['Day'] = self.NNfileroot+'/Daytime Models/'+self.NNID2imgs[curNN]
                else:
                    tempNNs['Night'] = self.NNfileroot+'/Nighttime Models/'+self.NNID2imgs[curNN]
            
            #Don't execute the rest of the function if a certain type of model is needed, but a model of that type wasn't selected.
            if nightRequired and tempNNs['Night'] == None:
                return
            elif dayRequired and tempNNs['Day'] == None:
                return   

            #Execute prediction process
            if len(curNNs) >= NNnumRequired: 
                #Load images into the images list
                for curComp in curComps:
                    images.append([self.DSID2imgs[gp(gp(gp(curComp)))],self.fileTree[self.DSID2imgs[curComp]]])
                    fileloc.append(self.DSID2imgs[curComp])
                #Execute the predict function based on the circumstances.
                if nightRequired and dayRequired: 
                    df,confs = NNM.predict(images, df, Export, dayModelDir = tempNNs['Day'], nightModelDir = tempNNs['Night'])
                elif nightRequired:
                    df,confs = NNM.predict(images, df, Export, nightModelDir = tempNNs['Night'])
                elif dayRequired:
                    df,confs = NNM.predict(images, df, Export, dayModelDir = tempNNs['Day'])

            def finishExport(): 
                thresholds = np.array(thresholdEntryBox.get().split(',')).astype(float)
                classes = []

                for index in range(thresholds.shape[0]+1):
                    if index == 0:
                        classes.append('0.0-'+str(thresholds[index]))
                    elif index != 0 and index+1 <= thresholds.shape[0]:
                        classes.append(str(thresholds[index-1])+'-'+str(thresholds[index]))
                    elif index+1 > thresholds.shape[0]:
                        classes.append(str(thresholds[index-1])+'-1.0')
                classes = np.array(classes)

                index = 0
                if thresholds.shape[0] <= 2 and np.where(thresholds > 1)[0].shape[0] == 0 and np.where(thresholds < 0)[0].shape[0] == 0:
                    for conf in confs:
                        I = np.argmin(np.abs(thresholds - conf[1]))
                        if thresholds[I] < conf[1]:
                            try:
                                os.mkdir(returnSettings['Export']+'/'+classes[I+1])
                            except:
                                p = 1
                            finally:
                                copyfile(fileloc[index],returnSettings['Export']+'/'+classes[I+1]+'/'+fileloc[index].split('/')[-1])
                        elif thresholds[I] >= conf[1]:
                            try:
                                os.mkdir(returnSettings['Export']+'/'+classes[I])
                            except:
                                p = 1
                            finally:
                                copyfile(fileloc[index],returnSettings['Export']+'/'+classes[I]+'/'+fileloc[index].split('/')[-1])  
                        index += 1
                    popup.destroy()
            
            def displayStats():
                #Display popup showing the histogram of the confidence values, and then ask the 
                #user to define his or her thresholds and subfolder names.
                popup = Tkinter.Toplevel()
                popup.wm_title('Post-Prediction Settings')
                popup.wm_geometry("500x250")
            
                fig = Figure(figsize=(5, 4), dpi=100)
                canvas = FigureCanvasTkAgg(fig,popup)
                p = fig.gca()
                m = np.array(confs)[:,1]
                p.hist(m)
                p.set_xlabel('Confidence Values for Presence of An Animal', fontsize = 15)
                p.set_ylabel('Frequency', fontsize = 15)
                canvas.draw()
                canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
                global thresholdEntryBox,classEntryBox
                label = Tkinter.Label(popup, text = 'Thresholds: ', justify = 'left')
                label.pack()
                thresholdEntryBox = Tkinter.Entry(popup)
                thresholdEntryBox.pack()

                okbutt = ttk.Button(popup, text="Okay", command=finishExport)
                okbutt.pack()
            try:
                if 'Results' in returnSettings:
                    df.to_csv(xceldir, encoding='utf-8', index=False)
            except:
                p = 1
            
            try:
                if 'Export' in returnSettings:
                    exportDir = returnSettings['Export']
                    displayStats()
            except:
                p = 1
            
            #Set the label text to update
            label['text'] = 'The neural network(s) is/are done predicting on the selected dataset components.'
            
            self.setupDatatreeview('') #reload the Datatreeview

    def getNNDirs(self):
        NNPathsNum = -1
        #Requires a specific folder architecture
        #Need to add flexibility 

        #Assumes that all the NN models belong to the model folder and that all the models have been categorized into folders.
        with os.scandir(self.NNfileroot) as dirs1:
            for dir1 in dirs1:
                if dir1.path.split('\\')[-1][0] != '.': #Makes sure the file isn't a hidden file
                    NNPathsNum += 1
                    #Create a new entry in the NNtreeview
                    id = self.NNtreeview.insert('','end',text = dir1.path.split('\\')[-1])
                    #Add the nn to the lookup
                    self.NNID2imgs[id] = dir1.path
                    self.NNimgs2ID[dir1.path] = id
                    #Add the path
                    self.NNPaths.append([dir1.path])
                    with os.scandir(dir1.path) as dirs2:
                        for dir2 in dirs2:
                            if dir2.path.split('\\')[-1].split('.')[-1] == 'model' or dir2.path.split('\\')[-1].split('.')[-1] == 'mat': #Makes the assumption that NN files must have the .model extension
                                info = dir2.stat()
                                d = datetime.datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                                idimp = self.NNtreeview.insert(id,'end',text = dir2.path.split('\\')[-1], values = (d,dir2.path))
                                self.NNimgs2ID[dir2.path.split('\\')[-1]] = idimp
                                self.NNID2imgs[idimp] = dir2.path.split('\\')[-1]
                                self.NNPaths[NNPathsNum].append(dir2.path)
                            else:
                                continue
  
    def LabelImgs(self):
        #Open the labeler
        self.openLabelImg()
        #Update the Datatreeview
        self.setupDatatreeview('')
         
    def openLabelImg(self):
        #Opens the labeler application
        app,win = labelImg.get_main_app()
        #Gets images from the Datatreeview panel
        curObjs = self.Datatreeview.selection()
        filepaths = []
        i = 0
        win.defaultSaveDir = dm.rootfp+'XMLs folder/'
        if len(curObjs) == 0:
            return
        for curObj in curObjs:
            filepaths.append(self.fileTree[self.DSID2imgs[curObj]].imgpath)
        win.mImgList = win.scanAllImages2(filepaths)
        for path in win.mImgList:
            item = QListWidgetItem(path)
            win.fileListWidget.addItem(item)
        app.exec_()
    
    #Used to update the GUI when something needs to be displayed before the end of the function.
    def show(self):
        self.parent.update()
        self.parent.deiconify()

    def makeDSDir(self):
        DSfile = filedialog.askdirectory()
        if DSfile != '' and self.currentDS == '':
            dm.createDSDir(DSfile)
            popup = Tkinter.Toplevel()
            popup.wm_title('DS was created')
            popup.wm_geometry("100x50")
            popup.resizable(False, False)
            label = Tkinter.Label(popup, text = 'DS was created', justify = 'left')
            label.grid(row = 1, column = 0)

            self.setupDatatreeview(DSfile+'\\Main.csv')

    #Removes Elements from the Datatreeview
    def deleteDatatreeview(self):
        global appname
        #Change the name back
        self.parent.title(appname)
        self.currentDS = ''
        #Clear
        self.fileTree = {}
        self.DSimgs2ID = {}
        self.DSID2imgs = {}

        #Get rid of the entries from the treeview object.
        for entry in self.Datatreeview.get_children():
            self.Datatreeview.delete(entry)

    def insert_data(self,piece):

        if piece in self.DSimgs2ID:
            return

        if not self.fileTree[piece].timegroup+'/'+piece.split('/')[-2] in self.DSimgs2ID:
            if not piece.split('/')[-2] in self.DSimgs2ID:
                superid = self.Datatreeview.insert('', 'end', text = piece.split('/')[-2],
                                        values=(), open = True)
                self.DSimgs2ID[piece.split('/')[-2]] = superid
                self.DSID2imgs[superid] = piece.split('/')[-2]
            else:
                superid = self.DSimgs2ID[piece.split('/')[-2]]
            superid = self.Datatreeview.insert(superid, 'end', text = self.fileTree[piece].timegroup,
                                    values=(), open = True)
            self.DSimgs2ID[self.fileTree[piece].timegroup+'/'+piece.split('/')[-2]] = superid
            self.DSID2imgs[superid] = self.fileTree[piece].timegroup+'/'+piece.split('/')[-2]
            id = self.Datatreeview.insert(superid, 'end', text = 'Day',
                                    values=(), open = True)
            self.DSID2imgs[str(id)] = self.fileTree[piece].timegroup+'/Day'
            self.DSimgs2ID[self.fileTree[piece].timegroup+'/'+piece.split('/')[-2]+'/Day'] = id
            id = self.Datatreeview.insert(superid, 'end', text = 'Night',
                                    values=(), open = True)
            self.DSID2imgs[str(id)] = self.fileTree[piece].timegroup+'/Night'
            self.DSimgs2ID[self.fileTree[piece].timegroup+'/'+piece.split('/')[-2]+'/Night'] = id
                                    
        if piece.split("\\")[-1].split(".")[-1].lower() == 'jpg' and not piece in self.DSimgs2ID:    
            if self.fileTree[piece].day:
                id = self.Datatreeview.insert(self.DSimgs2ID[self.fileTree[piece].timegroup+'/'+piece.split('/')[-2]+'/Day'], 'end', text = piece.split('/')[-1],
                                    values=('', self.fileTree[piece].nnLabeled(),self.fileTree[piece].humanLabeled()))
            else:
                id = self.Datatreeview.insert(self.DSimgs2ID[self.fileTree[piece].timegroup+'/'+piece.split('/')[-2]+'/Night'], 'end', text = piece.split('/')[-1],
                                    values=('', self.fileTree[piece].nnLabeled(),self.fileTree[piece].humanLabeled()))    
                

        self.DSID2imgs[str(id)] = piece
        self.DSimgs2ID[piece] = id
    
    def localizationpopup(self):
        #Opens the labeler application
        app,win = readonly.get_main_app()
        #Gets images from the Datatreeview panel
        curObjs = self.Datatreeview.selection()
        filepaths = []
        i = 0
        win.defaultSaveDir = dm.rootfp+'XMLs folder/'
        if len(curObjs) == 0:
            return
        for curObj in curObjs:
            filepaths.append(self.fileTree[self.DSID2imgs[curObj]].imgpath)
        win.mImgList = win.scanAllImages2(filepaths)
        for path in win.mImgList:
            item = QListWidgetItem(path)
            win.fileListWidget.addItem(item)
        app.exec_()

    def dirCopy2DS(self):
        if self.currentDS != '':
            DSfile = dm.rootfp+"imgsfolder"
            repository = "./Image Repository"
            for direct in os.listdir(repository):
                self.fileTree = dm.copyintoDS(repository+'/'+direct,self.fileTree)
                self.setupDatatreeview(DSfile)

    def CopyDSintoCurrDS(self):
        HDF5file = filedialog.askdirectory()

        if HDF5file != '':
            popup = Tkinter.Toplevel()
            popup.wm_title('Dataset is being added')
            popup.wm_geometry("100x50")
            popup.resizable(False, False)
            label = Tkinter.Label(popup, text = 'Dataset is being added', justify = 'left')
            label.grid(row = 1, column = 0)
            self.show()
            try:
                self.fileTree = dm.copyDSintoDS(HDF5file,self.fileTree)
                self.setupDatatreeview(HDF5file)
            except:
                p = 1
            label['text'] = 'Dataset was added'

    def openProperties(self,h,curitem = None):
        curitem = self.Datatreeview.focus()
        if curitem != '':
            curitemName = self.DSID2imgs[curitem]
            if curitemName.split('.')[-1].lower() == 'jpg':
                DSPopup(curitemName)
    
    def InquireDSFile(self):
        if self.currentDS == '':
            DSfile = filedialog.askopenfilename(initialdir = curdir, title = "Select CSV file",filetypes = (("CSV Files","*.csv"),("all files","*.csv")))
            if DSfile != '' and DSfile.split('.')[-1] == 'csv':
                try:
                    self.fileTree = dm.openDS(DSfile)
                    HDF5file = "\\".join(DSfile.split("/")[0:-1])+"\\imgsfolder"
                    self.setupDatatreeview(DSfile)
                except:
                    p = 1
        else:
            print("Can't open.")
    
    def setupDatatreeview(self,HDF5file):
        global appname
        self.parent.title(appname+' '+dm.rootfp+'Main.csv')
        #Get rid of the entries from the treeview object.
        self.DSimgs2ID = {}
        self.DSID2imgs = {}
        self.selectedDS = []
        for entry in self.Datatreeview.get_children():
            self.Datatreeview.delete(entry)
        for piece in self.fileTree:
            self.insert_data(piece)
        self.currentDS = HDF5file.split('\\')[-1]

def main():
    global gui #GUI
    root=Tkinter.Tk()
    gui = DatasetView(root)
    root.mainloop()

if __name__=="__main__":
    main()