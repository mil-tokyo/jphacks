import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *

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
        try:
            self.input = json_object["input"]
        except KeyError:
            self.input = None

        try:
            self.output = json_object["output"]
        except KeyError:
            self.output = None

    def __str__(self):
        return "{}:{},\n{}:{},\n{}:{}"\
          .format("name", self.name, "input", self.input, "output", self.output)
        
model_class_dict = {"KMeans" : "unsupervised", "SVC" : "classification", "LinearRegression" : "regression"}      
class Model(Object):
    def __init__(self, json_object):
        super(Model, self).__init__(json_object)
        self.model_type = str(json_object["model_type"])
        self.model_class = model_class_dict[self.model_type]
        self.params = json_object["params"]
        for k, v in self.params.items():
            if type(v) == unicode:
                self.params[k] = str(v)
                
        self.model = eval(str(self.model_type)+"(**self.params)")
        
    def calculate(self, input_data):
    #        if input_data["label"] is None:
        if self.model_class == "unsupervised":
            self.model.fit(input_data["data"][:, 1:])
        else:
            self.model.fit(input_data["data"][:, 1:], input_data["data"][:, 0])
            
        return {"data": input_data["data"], "model" : {"model_params" : self.model, "model_class" : self.model_class}}, self.output

class Visualizer(Object):
    def __init__(self, json_object):
        super(Visualizer, self).__init__(json_object)
        self.plot_range = [-10, 15, -10, 15]
        self.colors_list = ["b", "g", "r", "c", "m", "y", "k"]
        self.image_source = "./log/{}.png".format(self.name)
                
    def calculate(self, input_data):
        self.model = input_data["model"]["model_params"]
        mode = input_data["model"]["model_class"]
        print type(input_data["data"])
        self.data = input_data["data"][:, 1:]
        self.label = input_data["data"][:, 1]
        if mode == "unsupervised":
            self.label = self.model.predict(self.data)
        
        #self.label = input_data["data"]["label"]
        #print self.label
         #   if type(self.label[0]) == np.int64:
         #       mode = "classification"
         #   else:
         #       mode = "regression"
        #else:
        #    self.label = self.model.predict(self.data)
         #   mode = "unsupervised"
        plt.clf()
        self.plot_data(mode)
        self.plot_func(mode)
        plt.savefig(self.image_source)
        return {"name": self.name, "type": self.type, "data" : self.data.tolist(), "img_src" : self.image_source}, False


    def plot_func(self, mode):
        """ plot fucntions """
        if mode == "regression":
            axis_x = np.arange(-1, 2, 0.1)
            axis_y = [self.model.decision_function(np.array([x])) for x in axis_x]
            plt.plot(axis_x, axis_y, "-"+self.colors_list[-1])

        elif mode == "classification":
            axis_x = np.arange(-10, 10)
            axis_y = -(self.model.intercept_[0] + self.model.coef_[0, 0] * axis_x) / self.model.coef_[0, 1]
            plt.plot(axis_x, axis_y, "-"+self.colors_list[-1])

    def plot_data(self, mode):
        """plot data """
        n_labels = max(self.label) + 1
        if mode == "classification" or mode == "unsupervised":
            plt.axis(self.plot_range)
            for l in range(n_labels):
                x_plot = self.data[self.label == l, :]
                plt.plot(x_plot[:, 0], x_plot[:, 1], "o"+self.colors_list[l])
            plt.legend(range(n_labels), "lower right")

        else:
            plt.plot(self.data, self.label, "o")

        
class Data(Object):
    def __init__(self, json_object):
        super(Data, self).__init__(json_object)
        self.data = np.array(json_object["data"]["data"])
 #       try:
#        self.label = np.array(json_object["data"]["label"])
#        except KeyError:
#            self.label = None
            
    def calculate(self, input_data):
        return {"data": self.data}, self.output
