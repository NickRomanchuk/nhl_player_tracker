import yaml
import os             
import matplotlib.pyplot as plt

class DataVisualizer():
    def __init__(self, path):
        # Get class names
        with open(path + '/data.yaml', 'r') as file:
            config = yaml.safe_load(file)    
        self.names = config['names']

        # Collect bounding box data from the dataset
        self.data = {}
        for dataset in ['train', 'valid']:
            self.collect_data(path, dataset)

    def collect_data(self, path, type):
        # Initialize dictionaries
        counts, size_data = {}, {'width':{}, 'length':{}, 'area':{}, 'location':{}}
        for name in self.names:
            for key in size_data.keys():
                size_data[key][name] = []
            counts[name] = []

        # Loop over label files for each image 
        label_path = path + '/' + type + "/labels/"
        for fileName in os.listdir(label_path):
            file = open(label_path + fileName, "r")
            
            # Loop over labels in the image
            count_in_image = {}
            for line in file:
                cls_idx, x_center, y_center, width, length = line.split(" ")
                name = self.names[int(cls_idx)]

                size_data['width'][name].append(float(width))
                size_data['length'][name].append(float(length))
                size_data['area'][name].append(float(width) * float(length))
                size_data['location'][name].append((float(x_center), float(y_center)))
                count_in_image[name] = 1 + count_in_image.get(name, 0)
            
            for name in self.names:
              counts[name].append(count_in_image.get(name, 0))
        
        # Store label information
        self.data[type] = {'size':size_data, 'count':counts}

    def plot_box_size(self, type='train'):
        # Get data for boxplot sizes
        data = self.data[type]['size']

        # Plot the figure
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle('Bounding Boxes Shapes')
        ax1.boxplot(data['width'].values(), labels=data['width'].keys())
        ax1.set_title('Width')
        ax2.boxplot(data['length'].values(), labels=data['length'].keys())
        ax2.set_title('Length')
        ax3.boxplot(data['area'].values(), labels=data['area'].keys())
        ax3.set_title('Area')
        fig.tight_layout()
        plt.show()

    def plot_box_counts(self, type='train'):
        # Get data for boxplot counts
        data = self.data[type]['count']

        # Plot counts
        fig, axes = plt.subplots(len(self.names))
        fig.suptitle('Number of Occurences in each Frame')
        for name, ax in zip(self.names, axes):
             ax.hist(data[name])
             ax.set_title(name)
        fig.tight_layout()
        plt.show()
    
    def plot_box_location(self, type='train'):
        # Get data for boxplot sizes
        data = self.data[type]['size']
        
        # Plot box locations
        colors = list("rgbcmyk")
        for name, location in data['location'].items():
            x = [center[0] for center in location]
            y = [center[1] for center in location]
            plt.scatter(x, y, color=colors.pop())
        plt.title('Box Locations')
        plt.legend(data['location'].keys())
        plt.show()