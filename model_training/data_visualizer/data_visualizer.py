import yaml
import os             
import matplotlib.pyplot as plt
import cv2
import numpy as np

class DataVisualizer():
    def __init__(self, path):
        self.path = path + "/"

        with open(self.path + 'data.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        self.names = config['names']

    def collectBoxData(self, type):
        boxSizeData = {'width': {}, 'length': {}, 'area': {}, 'location': {}, 'red': {}, 'blue': {}, 'green': {}}
        allCounts = {}
        
        for name in self.names:
            boxSizeData['red'][name] = []
            boxSizeData['blue'][name] = []
            boxSizeData['green'][name] = []
            boxSizeData['width'][name] = []
            boxSizeData['length'][name] = []
            boxSizeData['area'][name] = []
            boxSizeData['location'][name] = []
            allCounts[name] = []

        labelPath = self.path + type + "/labels/"
        imagePath = self.path + type + "/images/"
        for fileName in os.listdir(labelPath):
            file = open(labelPath + fileName, "r")
            image = cv2.imread(imagePath + fileName[:-4] + '.jpg')
            print(fileName)
            counts = {}
            for line in file:
                lineList = line.split(" ")

                red, blue, green = self.getBoxColor(image, lineList)
                boxSizeData['red'][self.names[int(lineList[0])]].append(red)
                boxSizeData['blue'][self.names[int(lineList[0])]].append(blue)
                boxSizeData['green'][self.names[int(lineList[0])]].append(green)
                boxSizeData['width'][self.names[int(lineList[0])]].append(float(lineList[3]))
                boxSizeData['length'][self.names[int(lineList[0])]].append(float(lineList[4]))
                boxSizeData['area'][self.names[int(lineList[0])]].append(float(lineList[3]) * float(lineList[4]))
                boxSizeData['location'][self.names[int(lineList[0])]].append((float(lineList[1]), float(lineList[2])))
                counts[lineList[0]] = 1 + counts.get(lineList[0], 0)
            
            for i in range(len(self.names)):
              allCounts[self.names[i]].append(counts.get(str(i), 0))
        
        return boxSizeData, allCounts
    
    def getBoxColor(self, image, lineList):
        height, width, _ = image.shape

        bb_w = (float(lineList[3]) * width)
        bb_h = (float(lineList[4]) * height)
        x1 = (float(lineList[1]) * width) - (bb_w / 2)
        y1 = (float(lineList[2]) * height) - (bb_h / 2)
        x2 = x1 + bb_w
        y2 = y1 + bb_h

        bbox = image[int(y1):int(y2), int(x1):int(x2)]
        print(bbox.shape)
        return bbox.mean(axis=(0,1))

    def plotBoxSize(self, type='train'):
        boxData, _ = self.collectBoxData(type=type)

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle('Bounding Boxes Shapes')
        
        ax1.boxplot(boxData['width'].values(), labels=boxData['width'].keys())
        ax1.set_title('Width')
        ax2.boxplot(boxData['length'].values(), labels=boxData['length'].keys())
        ax2.set_title('Length')
        ax3.boxplot(boxData['area'].values(), labels=boxData['area'].keys())
        ax3.set_title('Area')
        fig.tight_layout()
        plt.show()

    def plotBoxColor(self, type='train'):
        boxData, _ = self.collectBoxData(type=type)

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle('Bounding Boxes Colors')

        ax1.boxplot(boxData['red'].values(), labels=boxData['red'].keys())
        ax1.set_title('Red')
        ax2.boxplot(boxData['blue'].values(), labels=boxData['blue'].keys())
        ax2.set_title('Blue')
        ax3.boxplot(boxData['green'].values(), labels=boxData['green'].keys())
        ax3.set_title('Green')
        fig.tight_layout()
        plt.show()

    def plotBoxCounts(self, type='train'):
        _, allCounts = self.collectBoxData(type=type)

        fig, axes = plt.subplots(len(self.names))
        fig.suptitle('Number of Occurences in each Frame')
        fig.tight_layout()

        for name, ax in zip(self.names, axes):
             ax.hist(allCounts[name])
             ax.set_title(name)

        plt.show()
    
    def plotBoxLocation(self, type='train'):
        boxData, _ = self.collectBoxData(type=type)
        print('Here')
        colors = list("rgbcmyk")
        
        for name, data in boxData['location'].items():
            x = [x[0] for x in data]
            y = [x[1] for x in data]
            plt.scatter(x, y, color=colors.pop())

        plt.legend(boxData['location'].keys())
        plt.show()