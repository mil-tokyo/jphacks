import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.cluster import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *

class Objects():
    def __init__(self, queue_id, json_objects):
        """ self.objects is dictionary
        key : name of module
        val : instance of class Data or Model or Visualizer
        """
        self.objects_dict = {}

        """ initialize objects dicitionary """
        for obj in json_objects:
            obj_type = obj['type']
            self.objects_dict[obj['name']] = eval(str(obj_type) + "(queue_id, obj)")
             
    def calculate(self):
        """where calculation will start"""
        start_obj_list = [obj for obj in self.objects_dict.values() if obj.type == "Data" or obj.type == "Extractor"]
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
    def __init__(self, queue_id, json_object):
        self.queue_id = queue_id
        self.type = json_object["type"]
        self.name = json_object["name"]
        self.input = json_object.get("input", None)
        self.output = json_object.get("output", None)

    def __str__(self):
        return "{}:{},\n{}:{},\n{}:{}"\
          .format("name", self.name, "input", self.input, "output", self.output)

        
model_class_dict = {"KMeans" : "unsupervised", "SVC" : "classification", "LinearSVC" : "image", "LinearRegression" : "regression"}      
class Model(Object):
    def __init__(self, queue_id, json_object):
        super(Model, self).__init__(queue_id, json_object)
        self.read_model_path = json_object.get("model_filename", False)
        if self.read_model_path:
            self.model_type = str(json_object["model_type"])
            self.model_class = json_object["model_class"]
            return 0
        self.model_type = str(json_object["model_type"])
        self.model_class = model_class_dict[self.model_type]
        self.params = json_object.get("params", False)
        for k, v in self.params.items():
            if type(v) == unicode:
                self.params[k] = str(v)
        self.model = eval(str(self.model_type)+"(**self.params)")

        
    def calculate(self, input_data):
        if self.read_model_path:
            return {"name" : self.name, "type" : self.type, "model_type" : self.model_type, "data": input_data["data"], "model" : {"model_filename" : self.read_model_path, "model_class" : self.model_class}}, self.output
            
        if self.model_class == "unsupervised":
            self.model.fit(input_data["data"][:, 1:])
      
        elif self.model_class == "image":
            self.model = joblib.load('./model/linSVM.pkl')
        else:
            self.model.fit(input_data["data"][:, 1:], input_data["data"][:, 0])

        model_filename = "./model/{}_{}.pkl".format(self.queue_id, self.model_type)
        joblib.dump(self.model, model_filename)
        return {"name" : self.name, "type" : self.type , "model_type" : self.model_type, "data": input_data["data"], "model" : {"model_filename" : model_filename, "model_class" : self.model_class}}, self.output

class Visualizer(Object):
    def __init__(self, queue_id, json_object):
        super(Visualizer, self).__init__(queue_id, json_object)
        self.colors_list = ["b", "r", "g", "c", "m", "k", "y"]
        self.image_source = "./log/{}_{}.png".format(self.queue_id, self.name)
                
    def calculate(self, input_data):
        model_filename = input_data.get("model", {"model_filename" : False})["model_filename"]            
        self.model = joblib.load(model_filename) if model_filename else "Data"
        model_type = input_data.get("model_type", False)
        mode = input_data.get("model", {"model_class" : False})["model_class"]
        if mode:
            if mode == "image":
                class_name = ['buddha', 'camera', 'euphonium', 'snoopy', 'water_lilly']
                pred_ind = self.model.predict(input_data["data"])
                pred_class = class_name[int(pred_ind[0])]
                return {"data" : input_data["data"], "name": self.name, "type": self.type, "predict_class" : pred_class }, False
            else:
                self.is_plot_data = True
                self.is_plot_func = True
        else:
            self.is_plot_data = True
            self.is_plot_func = False
            mode = 'data'
            
        self.data = input_data["data"][:, 1:]
        self.label = input_data["data"][:, 0]
        
        if mode == "unsupervised":
            self.label = self.model.predict(self.data)
            
        plt.clf()
        plt.title(self.model.__class__.__name__)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        
        if self.is_plot_data:
            self.plot_data(mode)
        if self.is_plot_func:
            self.plot_func(mode)
            
        plt.savefig(self.image_source)
        return {"name": self.name, "type": self.type, "data" : self.data, "img_src" : self.image_source, "model" : {"model_filename" : model_filename, "model_type" : model_type, "model_class" : mode}}, False

    def plot_func(self, mode):
        """ plot fucntions """
        x_min, x_max = np.min(self.data[:, 0]), np.max(self.data[:, 0])
        
        if mode == "regression":
            axis_x = np.arange(x_min, x_max, 0.05)
            axis_y = [self.model.decision_function(np.array([x])) for x in axis_x]
            plt.plot(axis_x, axis_y, "-"+self.colors_list[1], linewidth=3)

        elif mode == "classification":
            axis_x = np.arange(x_min, x_max, 0.05)
            axis_y = -(self.model.intercept_[0] + self.model.coef_[0, 0] * axis_x) / self.model.coef_[0, 1]
            plt.plot(axis_x, axis_y, "-"+self.colors_list[-1], linewidth=3)

    def plot_data(self, mode):
        """plot data """
        n_labels = int(max(self.label)) + 1
        if mode == "classification" or mode == "unsupervised":
            for l in range(n_labels):
                x_plot = self.data[self.label == l, :]
                plt.plot(x_plot[:, 0], x_plot[:, 1], "o"+self.colors_list[l])
            plt.legend(range(n_labels), "lower right")
            
        else:
            if self.data.shape[1] == 1:
                plt.plot(self.data[:, 0], self.label, "o")
            elif self.data.shape[1] == 2:
                plt.plot(self.data[:, 0], self.data[:, 1], "o")
            else:
                raise ValueError, "dimension of data must be 1 or 2 for visualization"
            
class Data(Object):
    def __init__(self, queue_id, json_object):
        super(Data, self).__init__(queue_id, json_object)
        if type(json_object["data"]["data"]) == list:
            self.source_type = "array"
            self.data = np.array(json_object["data"]["data"])
        else:
            self.source_type = "image_path"
            self.img_src = json_object["data"]["data"]
            im = Image.open(self.img_src)
            im = ImageOps.grayscale(im)
            self.image = np.array(im.resize((200, 300)))
            self.extractor_type = "Hog"

    def calculate(self, input_data):
        if self.source_type == "array":
            return {"name" : self.name, "type" : self.type, "data": self.data}, self.output

        if self.source_type == "image_path":
            self.feature = hog(self.image, orientations=8, pixels_per_cell=(16, 16),\
                        cells_per_block=(1, 1), visualise=False)
            return {"name" : self.name, "type" : self.type, "data" : self.feature}, self.output
    
        

