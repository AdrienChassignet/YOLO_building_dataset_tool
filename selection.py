from __future__ import division
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import sys
import glob
import random

import argparse

import cv2
from ctypes import *
import numpy as np

TMP_FOLDER = './tmp/'

MAIN_COLORS = ['skyblue', 'pink', 'green', 'gold', 'blue', 'cyan']
#MAIN_COLORS = ['darkolivegreen', 'darkseagreen', 'darkorange', 'darkslategrey', 'darkturquoise', 'darkgreen', 'darkviolet', 'darkgray', 'darkmagenta', 'darkblue', 'darkkhaki','darkcyan', 'darkred',  'darksalmon', 'darkslategray', 'darkgoldenrod', 'darkgrey', 'darkslateblue', 'darkorchid','skyblue','yellow','orange','red','pink','violet','green','brown','gold','Olive','Maroon', 'blue', 'cyan', 'black','olivedrab', 'lightcyan', 'silver']

# image sizes for the examples
SIZE = 418, 418

classes = ['Puddle']

try:
    with open('classes.txt','r') as cls:
        classes = cls.readlines()
    classes = [cls.strip() for cls in classes]
except IOError as io:
    print("[ERROR] Please create classes.txt and put your all classes")
    sys.exit(1)
COLORS = random.sample(set(MAIN_COLORS), len(classes))

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

class LabelTool():
    def __init__(self, master, args):
        # set up the main frame
        self.curimg_h = 0
        self.curimg_w = 0
        self.cur_cls_id = -1
        self.parent = master
        self.parent.title("Yolo Annotation Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.videoDir = ''
        self.videoList= []
        self.curVideo = 0
        self.tmpDir = './tmp/'
        self.imageList= []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        #darknet variables
        self.yolo = args.yolo
        if self.yolo:
            self.metaMain = None
            self.netMain = None
            self.altNames = None
            self.instantiateDarknet(args.config, args.weight, args.meta)

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.bboxListCls = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Video Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.focus_set()
        self.entry.bind('<Return>', self.loadEntry)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
        self.ldBtn.grid(row = 0, column = 2, sticky = W+E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Escape> to cancel current bbox
        self.parent.bind("s", self.saveImage)
        self.parent.bind("<Left>", self.prevImage)
        self.parent.bind("<Right>", self.nextImage)
        self.parent.bind("a", self.prevImage) # press 'a' to go backforward
        self.parent.bind("d", self.nextImage) # press 'd' to go forward
        self.parent.bind("f", self.skip10Image)
        self.parent.bind("g", self.nextVideo)
        self.parent.bind("x", self.clearBBox)
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
        self.tkvar = StringVar(self.parent)
        self.cur_cls_id = 0
        self.tkvar.set(classes[0]) # set the default option
        self.popupMenu = OptionMenu(self.frame, self.tkvar, *classes,command = self.change_dropdown)
        self.popupMenu.grid(row = 1, column =2, sticky = E+S)
        self.chooselbl = Label(self.frame, text = 'Choose Class:')
        self.chooselbl.grid(row = 1, column = 2, sticky = W+S)
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 2, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 30, height = 12)
        self.listbox.grid(row = 3, column = 2, sticky = N)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 4, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll (X)', command = self.clearBBox)
        self.btnClear.grid(row = 5, column = 2, sticky = W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 6, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='Prev (A)', width = 6, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.saveBtn = Button(self.ctrPanel, text='Save (S)', width = 6, command = self.saveImage)
        self.saveBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Next (D)', width = 6, command = self.nextImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.skipBtn = Button(self.ctrPanel, text='Next10 (F)', width = 6, command = self.skip10Image)
        self.skipBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.passBtn = Button(self.ctrPanel, text='Pass (G)', width = 6, command = self.nextVideo)
        self.passBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)

        # create keyboard shortcuts
        #master.bind('d', self.nextImage)
        #master.bind('a', self.prevImage)

    def loadEntry(self,event):
        self.loadDir()

    def loadDir(self, dbg = False):
        if not dbg:
            try:
                s = self.entry.get()
                self.parent.focus()
                self.category = s
            except ValueError as ve:
                messagebox.showerror("Error!", message = "The folder should be numbers")
                return
        if not os.path.isdir('./Videos/%s' % self.category):
           messagebox.showerror("Error!", message = "The specified dir doesn't exist!")
           return
        # get image list
        self.videoDir = os.path.join(r'./Videos', '%s' %(self.category))
        self.videoList = glob.glob(os.path.join(self.videoDir, '*.avi'))
        if len(self.videoList) == 0:
            print('No .avi videos found in the specified dir!')
            messagebox.showerror("Error!", message = "No .avi videos found in the specified dir!")
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.curVideo = 0
        self.total = self.store_all_frames(self.videoList[self.curVideo])

         # set up output dir
        if not os.path.exists('./Output'):
            os.mkdir('./Output')
        self.outDir = os.path.join(r'./Output', '%s' %(self.category))
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
        self.loadImage()

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        self.curimg_w, self.curimg_h = self.img.size
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # load labels
        self.clearBBox()
        # self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        self.imagename = os.path.splitext(os.path.basename(imagepath))[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    yolo_data = line.strip().split()
                    tmp = self.deconvert(yolo_data[1:])
                    self.bboxList.append(tuple(tmp))
                    self.bboxListCls.append(yolo_data[0])
                    tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                            tmp[2], tmp[3], \
                                                            width = 2, \
                                                            outline = COLORS[int(yolo_data[0])])
                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%d, %d) -> (%d, %d) -> (%s)' %(tmp[0], tmp[1], tmp[2], tmp[3], classes[int(yolo_data[0])]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[int(yolo_data[0])])
        elif self.yolo:
            detections = darknet.detect(self.netMain, self.metaMain, os.path.abspath(imagepath).encode("ascii"), debug=False)
            for detection in detections:
                label = detection[0].decode('utf-8')
                bounds = convertBack(detection[2][0], detection[2][1], detection[2][2], detection[2][3])
                self.bboxList.append(tuple(bounds))
                self.bboxListCls.append(classes.index(label))
                tmpId = self.mainPanel.create_rectangle(bounds[0], bounds[1], bounds[2], bounds[3], width=2, outline = COLORS[classes.index(label)])
                self.bboxIdList.append(tmpId)
                self.listbox.insert(END, '(%d, %d) -> (%d, %d) -> (%s)' %(bounds[0], bounds[1], bounds[2], bounds[3], label))
                self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[classes.index(label)])

    def saveImage(self, event=None):
        with open(self.labelfilename, 'w') as f:
            for bbox,bboxcls in zip(self.bboxList,self.bboxListCls):
                xmin,ymin,xmax,ymax = bbox
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = self.convert((self.curimg_w,self.curimg_h), b)
                f.write(str(bboxcls) + " " + " ".join([str(a) for a in bb]) + '\n')
        os.system('cp {0} {1}'.format(os.path.join(self.tmpDir, self.imagename+'.jpg'), self.outDir))
        print('Image No. %d saved ({0})'.format(self.imagename) %(self.cur))


    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxListCls.append(self.cur_cls_id)
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%d, %d) -> (%d, %d) -> (%s)' %(x1, y1, x2, y2, classes[self.cur_cls_id]))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[self.cur_cls_id])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text = 'x: %03d, y: %03d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = COLORS[self.cur_cls_id])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        print(self.bboxListCls,idx)
        self.bboxListCls.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.bboxListCls = []

    def prevImage(self, event = None):
        #self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()
        else:
            messagebox.showerror("Information!", message = "This is first image")

    def nextImage(self, event = None, skip=1):
        #self.saveImage()
        if self.cur <= self.total - skip:
            self.cur += skip
            self.loadImage()
        elif self.cur == self.total:
            messagebox.showerror("Information!", message = "All images annotated")
        else:
            messagebox.showerror("Information!", message = "Cannot skip more")
        
    def skip10Image(self, event=None):
        self.nextImage(skip=10)

    def nextVideo(self, event = None):
        if self.curVideo < len(self.videoList)-1:
            self.clean_tmp_folder()
            self.cur = 1
            self.curVideo += 1
            self.total = self.store_all_frames(self.videoList[self.curVideo])
            self.loadImage()
        else:
            messagebox.showerror("Information!", message = "All videos processed")

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            #self.saveImage()
            self.cur = idx
            self.loadImage()

    def change_dropdown(self,*args):
        cur_cls = self.tkvar.get()
        self.cur_cls_id = classes.index(cur_cls)

    def store_all_frames(self, vidPath):
        vidName = os.path.split(os.path.splitext(vidPath)[0])[-1]
        vidcap = cv2.VideoCapture(vidPath)
        self.imageList = []
        count = 1
        success, frame = vidcap.read()
        while success:
            imgName = self.tmpDir + vidName +"_frame{0}.jpg".format(count)
            self.imageList.append(imgName)
            cv2.imwrite(imgName, frame)
            success,frame = vidcap.read()
            if success: 
                count += 1
        vidcap.release()
        print('%d images loaded from %s' %(count, vidName))
        return count

    def clean_tmp_folder(self):
        for f in os.listdir(self.tmpDir):
            os.remove(os.path.join(self.tmpDir, f))

    def convert(self,size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def deconvert(self,annbox):
        ox = float(annbox[0])
        oy = float(annbox[1])
        ow = float(annbox[2])
        oh = float(annbox[3])
        x = ox*self.curimg_w
        y = oy*self.curimg_h
        w = ow*self.curimg_w
        h = oh*self.curimg_h
        xmax = (((2*x)+w)/2)
        xmin = xmax-w
        ymax = (((2*y)+h)/2)
        ymin = ymax-h
        return [int(xmin),int(ymin),int(xmax),int(ymax)]

    def instantiateDarknet(self, configPath, weightPath, metaPath):
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to select and label frames from videos')
    parser.add_argument('--yolo', action='store_true', help='Indicate if you want to use existing YOLO model to pre-label frames. Not enable by default')
    parser.add_argument('--darknet-path', default='../darknet', help='Path to the darknet repo')
    parser.add_argument('--config', default='../darknet/dataset_1000/yolov4-puddles.cfg', help='Path of the yolo config file')
    parser.add_argument('--weight', default='../darknet/dataset_1000/backup/fold0/yolov4-puddles_best.weights', help='Path of the yolo weights file')
    parser.add_argument('--meta', default='../darknet/dataset_1000/obj0.data', help='Path of the yolo meta file')
    args = parser.parse_args()

    if args.yolo:
        sys.path.append(os.path.abspath(args.darknet_path))
        import darknet

    root = Tk()
    tool = LabelTool(root, args)
    root.resizable(width =  True, height = True)
    root.mainloop()
    tool.clean_tmp_folder()
