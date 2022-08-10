import os
import datetime
import tensorflow as tf

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
        self.plot.model(f"{model_path}/{self.now}.png")
        
    def save_weights(self):
        model_path = f"{self.root_dir}/logs/{self.filename}/{self.net.model_type}/weights"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        self.model.save_weights(f"{model_path}/{self.now}.h5")
        
    def load_model(self, model):
        self.model = model
        
    def add_callback(self, callback):
        self.callbacks.append(callback)
        
def _tf_warnings(state='2'):
    """
    Switch state of tensorflow warnings. 
    
    In detail:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed

    Parameters
    ----------
    state : bool or str, optional
        Level of tensorflow logging state. The default is '2'.

    Returns
    -------
    None.

    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = state
    # tf.autograph.set_verbosity(10)




