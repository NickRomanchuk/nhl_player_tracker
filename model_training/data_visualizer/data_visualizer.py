import yaml
import os             
import matplotlib.pyplot as plt

class DataVisualizer():
    def __init__(self, path):
        self.path = path + "/"

        with open(self.path + 'data.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        self.names = config['names']

    def collectBoxData(self, type):
        boxSizeData = {'width': {}, 'length': {}, 'area': {}, 'location': {}}
        allCounts = {}
        
        for name in self.names:
            boxSizeData['width'][name] = []
            boxSizeData['length'][name] = []
            boxSizeData['area'][name] = []
            boxSizeData['location'][name] = []
            allCounts[name] = []

        filePath = self.path + type + "/labels/"
        for fileName in os.listdir(filePath):
            file = open(filePath + fileName, "r")

            counts = {}
            for line in file:
                lineList = line.split(" ")

                boxSizeData['width'][self.names[int(lineList[0])]].append(float(lineList[3]))
                boxSizeData['length'][self.names[int(lineList[0])]].append(float(lineList[4]))
                boxSizeData['area'][self.names[int(lineList[0])]].append(float(lineList[3]) * float(lineList[4]))
                boxSizeData['location'][self.names[int(lineList[0])]].append((float(lineList[1]), float(lineList[2])))
                counts[lineList[0]] = 1 + counts.get(lineList[0], 0)
            
            for i in range(len(self.names)):
              allCounts[self.names[i]].append(counts.get(str(i), 0))
        
        return boxSizeData, allCounts
    
        
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

        colors = list("rgbcmyk")
        
        for name, data in boxData['location'].items():
            x = [x[0] for x in data]
            y = [x[1] for x in data]
            plt.scatter(x, y, color=colors.pop())

        plt.legend(boxData['location'].keys())
        plt.show()