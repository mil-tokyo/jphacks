import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Objects():
    def __init__(self, json_objects):
        self.objects = {}
        for obj in json_objects:
            obj_type = obj['type']        
            if obj_type == "data":
                self.objects[obj['name']] = Data(obj)
            elif obj_type == "model":
                self.objects[obj['name']] = Model(obj)
            elif obj_type == "visualize":
                self.objects[obj['name']] = Visualizer(obj)


                       
    def calculate(self):
        input_data = None

        ########
        #######
        obj = self.objects['source1']
        ######
        ######
                
        output_obj = True
        while 1:
            print obj
            result, output_obj = obj.calculate(input_data)
            if not output_obj:
                break
            obj = self.objects[output_obj]
            input_data = result

        return result
    
    def __str__(self):
        return "\n\n".join([str(obj) for obj in self.objects])

class Object(object):
    def __init__(self, json_object):
        self.name = json_object["name"]
        self.input = json_object["input"]
        self.output = json_object["output"]

    def __str__(self):
        return "{}:{},\n{}:{},\n{}:{}"\
          .format("name", self.name, "input", self.input, "output", self.output)
        
          
class Model(Object):
    def __init__(self, json_object):
        super(Model, self).__init__(json_object)
        self.model = KMeans(n_clusters=2)
        
    def calculate(self, input_data):
        print input_data
        print input_data.shape
        self.model.fit(input_data)
        return {"data":input_data, "model" : self.model}, self.output 

    
class Visualizer(Object):
    def __init__(self, json_object):
        super(Visualizer, self).__init__(json_object)
        self.plot_range = [-10, 15, -10, 15]
        self.colors_list = ["r", "g"]
        self.image_source = "./log/kmean.png"
        
    def calculate(self, input_data):
        self.data, self.model = input_data["data"], input_data["model"]
        y = self.model.predict(self.data)
        n_labels = max(y) + 1
        plt.axis(self.plot_range)
        
        for label in range(n_labels):
            x_plot = self.data[y == label, :]
            print x_plot
            plt.plot(x_plot[:, 0], x_plot[:, 1], "o"+self.colors_list[label])
        plt.savefig(self.image_source)
        
        return {"data" : self.data.tolist(), "img_src" : self.image_source}, False
        
class Data(Object):
    def __init__(self, json_object):
        super(Data, self).__init__(json_object)
        print json_object["data"][0]
        self.data = np.array(json_object["data"])
    
    def calculate(self, input_data):    
        return self.data, self.output

if __name__ == "__main__":
    
    decoded_json = [{'type' : 'data',
                    'name' : 'source_kane',
                    'input' : '',
                    'output' : 'k-means_kane',
                    'data' : '[[0, 0.1], [0.01, 0.01], [-0.01, -0.01], [0, 0.1],[1.11, 0.99], [0.99, 0.99],[1, 1.1], [1.1, 1]]' 
                    },
                    {'type' : 'model',
                    'name' : 'k-means_kane',
                    'input' : 'source_kane',
                    'output' : 'vis_kane',
                    'model_type' : 'k-means'
                    },
                    {'type' : 'visualize',
                    'name' : 'vis_kane',
                    'input' : 'k-means_kane',
                    'output' : ''
                    }
                    ]

    print decoded_json
    a= Objects(decoded_json)
    a.calculate()


        
