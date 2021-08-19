import os
import datetime
from . import callbacks, plot
import numpy as np
import itertools

class LogDir:
    def __init__(self, filename, net = None, callbacks = None):
        self.root_dir = "/".join(__file__.split("\\")[:-3])
        if not net:
            #try load filefolder
            pass
        if callbacks:
            if not isinstance(callbacks, list):
                callbacks = list([callbacks])
        self.callbacks = callbacks
        self.net = net
        self.model = self.net.model
        self.filename = filename
        self.now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(f"{self.root_dir}/logs/{self.filename}"):
            os.makedirs(f"{self.root_dir}/logs/{self.filename}")
    
    def log(self):
        self.save_model()
        self.save_weights()
        return
    
    def save_model(self):    
        model_path = f"{self.root_dir}/logs/{self.filename}/{self.net.model_type}/model"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        model_json = self.model.to_json()
        with open(f"{model_path}/{self.now}.json", "w") as json_file:
            json_file.write(model_json)
        plot.model(self.model, f"{model_path}/{self.now}.png")
        
    def save_weights(self):
        model_path = f"{self.root_dir}/logs/{self.filename}/{self.net.model_type}/weights"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        self.model.save_weights(f"{model_path}/{self.now}.h5")
        
    def load_model(self, model):
        self.model = model
        
    def add_callback(self, callback):
        self.callbacks.append(callback)


