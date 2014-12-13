import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Objects():
    def __init__(self, json_objects):
        """ self.objects is dictionary
        key : name of module
        val : instance of class Data or Model or Visualizer
        """
        self.objects_dict = {}

        """ initialize objects dicitionary """
        for obj in json_objects:
            obj_type = obj['type']
            self.objects_dict[obj['name']] = eval(str(obj_type) + "(obj)")
            
                       
    def calculate(self):
        """where calculation will start"""
        start_obj_list = [obj for obj in self.objects_dict.values() if obj.type == "Data"]
        results = []
        for obj in start_obj_list:
            """initialize"""
            input_data = None
            output_obj_name = True

            while 1:
                result, output_obj_name = obj.calculate(input_data)

                """ if calculation reach the end module """
                if not output_obj_name:
                    results.append(result)
                    break
                
                obj = self.objects_dict[output_obj_name]
                input_data = result

        return results
    
    def __str__(self):
        return "\n\n".join([str(obj) for obj in self.objects_dict])

class Object(object):
    def __init__(self, json_object):
        self.type = json_object["type"]
        self.name = json_object["name"]
        self.input = json_object["input"]
        self.output = json_object["output"]

    def __str__(self):
        return "{}:{},\n{}:{},\n{}:{}"\
          .format("name", self.name, "input", self.input, "output", self.output)
        
          
class Model(Object):
    def __init__(self, json_object):
        super(Model, self).__init__(json_object)
        model_type = json_object["model_type"]
        params = json_object["params"]
        self.model = eval(str(model_type)+"(**params)")
        
    def calculate(self, input_data):
        self.model.fit(input_data)
        return {"data":input_data, "model" : self.model}, self.output 

class Visualizer(Object):
    def __init__(self, json_object):
        super(Visualizer, self).__init__(json_object)
        self.plot_range = [-10, 15, -10, 15]
        self.colors_list = ["r", "g"]
        self.image_source = "./log/{}.png".format(self.name)
                
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


        
