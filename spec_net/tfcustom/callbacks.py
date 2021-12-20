# TODOs:
# - Implementation of the ability for using callback in conv layers
import numpy as np
from tensorflow.keras.callbacks import Callback # inherits from keras callbacks

# callback information variables in different containers:
from .utils._callback import CallbackLog, CallbackParams, CallbackTrigger
from ..utils.syntax import add_params
from ..preprocess import dataformat as df
            
class FeatureSelection(Callback):
    def __init__(self, 
                 validation_data,
                 layer_name, 
                 n_features=None, 
                 callback=None):
        """
        This class is a customized callback function to iteratively delete all 
        the unnecassary input nodes (n_in) of a 'LinearPass' layer. The number 
        of nodes is reduced successively with respect to the optimum of the 
        given metric. 
        
        The reduction or killing process is triggered whenever the threshold 
        is surpassed for a given interval (e.g. 90 % accuracy for at least 15 
        epochs).
        
        The reduction function itself is an exponential or linear approximation 
        towards the desired number of leftover nodes (n_features):
            
        Exponential:
            n_features = n_in * (1-r_p)**n_i
        
        Linear:
            n_features = n_in - n_p*n_i
        
        At each step there is a fixed percentage r_p or number n_p of nodes 
        with irrelevant data that is to be deleted.

        Parameters
        ----------
        layer_name : str
            Name of layer with class type 'LinearPass'. Specifies the layer 
            where irrelevant nodes are being deleted.
        n_features : float, optional
            Number of leftover optimized relevant nodes. If 'None', the node 
            killing lasts until threshold cannot be surpassed anymore. The 
            default is None.
        callback : dict
            A dictionary with all the feature selection callback parameters.
            Use get_callback_keys() method to get an overview on the valid 
            keywords. The default is None.
        
        Returns
        -------
        None.

        """
        super().__init__()
        self.set_params(params=callback)
        
        self.validation_data = self.get_validation_subset(validation_data,
                                                          self.params.n_samples)
        self.I = self.get_identity(self.params.loocv)
        
        self.layer_name = layer_name
        self.n_features = n_features
       
    def __repr__(self):
        return 'Feature Selection Callback'
    
    def set_params(self, params):
        """
        Overwrites the original set_params() in Keras' Callback class and sets
        a parameter class for this specific callback, where all the necessary
        parameters are saved in the class object of type CallbackParams.

        Parameters
        ----------
        params : dict
            The callback dictionary with all the paramter information for the
            instantiation of CallbackParams class.

        Returns
        -------
        None.

        """
        if params is None:
            self.params = CallbackParams()
        
        elif 'callback' in params.keys():
            self.params = CallbackParams(**params['callback'])
        
        else:
            return
        
    def on_epoch_end(self, epoch, logs={}):
        """
        Actual callback function that is called after every epoch. Its 
        procedure is a s follows:
            1. Initialize: 
                Store variables in self.logs (log for all iterations of the
                callback) and self.trigger (updates of each trigger variable
                after each epoch).
            2. Update of stopping criteria:
                Decision whether to stop training and feature selection or not.
            3. Update of weights:
                If callback is not triggered, the algorithm will not update
                anything. If it is triggered, it chooses the most 
                uninformative features to be pruned.

        Parameters
        ----------
        epoch : int
            The current training epoch.
        logs : dict, optional
            A dictionary of the current epoch's metric. The default is {}.

        Returns
        -------
        None.

        """
        if not hasattr(self, 'log'):
            self.initialize(logs)
        
        self._first_condition(logs)
        
        self._update_weights(epoch, logs)
        
        self._update_stopping_criteria()
        
        self._stop_model(epoch, logs)
        
    # internal functions to generate the validation data set for the 
    # feature evaluation task
    def get_identity(self, inv):
        """
        Provides an identity matrix or bitwise inverted identity matrix, 
        respectively. The matrix also considers the already pruned nodes. The 
        inverted identity matrix corresponds to the LOOCV approach and is just
        masking one node at a time. The uninverted matrix behaves like a mask 
        that only allows one node at each validation step at all.

        Parameters
        ----------
        inv : bool
            Defines whether the identity matrix has to be inverted bitwise. If 
            True, it is inverted.

        Returns
        -------
        I : ndarray
            The resulting identity matrix.

        """
        I = np.identity(self.validation_data[0].shape[1])
        
        if inv: # bitwise inversion
            I = np.ones(I.shape) - I
        
        return I
    
    def get_validation_subset(self, validation_data, n_samples):
        """
        Extracts an arbitrary subset from the validation data as an input for
        the feature selection algorithm.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples per class that are evaluated by the feature
            selection algorithm.

        Returns
        -------
        validation_data : tuple
            The validation data array and the corresponding labels.
        n_samples : int
            The validation subset label array.

        """
        data, labels = validation_data
        
        if n_samples:
            classes = np.unique(labels)
            indices = []
            for c in classes:
                indices.append(np.argwhere(labels==c).flatten()[:n_samples])
            indices = np.array(indices).flatten()
            
            data, labels = data[indices], labels[indices]
        
        return (data, labels)
    
    def map_validation_data(self):
        data, labels = self.validation_data
        n_samples, n_in = data.shape
        
        # one-hot-encoding function:
        labels_one_hot = df.one_hot(labels)
        
        n_c = labels.shape[1]
        
        # create new data containers for validation data and labels
        validation_data = np.empty([n_samples * self.log.n_features[-1],
                                    n_in])
        validation_labels = np.zeros([n_samples * self.log.n_features[-1],
                                      n_c])
        
        for i, idx in enumerate(self.log.index[-1]):
            lb = i * n_samples 
            ub = (i + 1) * n_samples
            validation_data[lb : ub] = data * self.I[idx]
            validation_labels[lb : ub] = labels_one_hot
        
        return validation_data, validation_labels
        
        
    def initialize(self, logs):
        """
        Instantiates the remaining callback containers: Log and Trigger. 

        Parameters
        ----------
        logs : dict
            Contains information on the training results (accuracy and loss 
            values).

        Returns
        -------
        None.

        """
        self.log = CallbackLog(self.params, self.model, self.layer_name)
        self.log.initialize(logs)
        
        self.trigger = CallbackTrigger(self.params, self.model, 
                                       self.layer_name)
        self.trigger.initialize(logs)
        
        self.n_classes = len(np.unique(self.validation_data[1]))
        self.n_in = self.log.n_features[0]
    
    def get_callback_keys(self):
        """
        Returns a list of possible callback parameter keywords. 

        Returns
        -------
        keys : list
            List of possible callback parameter keywords.

        """
        keys = self.params.__dict__.keys()
        return keys
    
    # check functions: accuracy or loss based feature selection callback?
    def _first_condition(self, logs):
        """
        Checks whether the first condition is met. There are different possible
        conditions depending on the metric (loss or accuracy).

        Parameters
        ----------
        logs : dict
            Contains information on the training results (accuracy and loss 
            values).

        Returns
        -------
        None.

        """
        a = self.trigger.grad
        b = self.trigger.thresh
        c = self.trigger.gradient
        d = self.trigger.outcome
        
        if self.params.metric in ['accuracy', 'val_accuracy']:
            b, d = d, b
        
        if a and b:
            if c <= a and d <= b:
                self.trigger.update_hit(logs)
            else:
                self.trigger.update_miss(logs)
        
        elif a and not b:
            if c <= a:
                self.trigger.update_hit(logs)
            else:
                self.trigger.update_miss(logs)
        
        elif not a and b:
            if d <= b:
                self.trigger.update_hit(logs)
            else:
                self.trigger.update_miss(logs)
        else:
            self.trigger.update_miss(logs)
    
    #check function: stop model?
    def _stop_model(self, epoch, logs):
        """
        Checks whether stopping conditions are met.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        logs : dict
            Contains information on the training results (accuracy and loss 
            values).

        Returns
        -------
        None.

        """
        
        if self.trigger.criterion_max and self.trigger.criterion_features:
            weights, loss = self._prune_weights(epoch)
            self.log.update(epoch, loss)
            self.trigger.converged = True
            self.model.stop_training = True
            print(f"Epoch {epoch} - Successful Feature Selection:\n"
                  f"Stopped training with '{self.params.metric}' of "
                  f"{logs[self.params.metric]} using "
                  f"{self.log.n_features[-1]} nodes as input features.")
        
        elif self.trigger.criterion_max:
            weights, loss = self._prune_weights(epoch)
            self.log.update(epoch, loss)
            self.model.stop_training = True
            print(f"Epoch {epoch} - Non Convergent:\n"
                  "The optimizer did not converge. Please adjust the feature "
                  "selection and/or model parameters and try again.\n\nStopped "
                  f"training with '{self.params.metric}' of "
                  f"{logs[self.params.metric]} using "
                  f"{self.log.n_features[-1]} nodes as input features.")
    
    #update functions: weights and stopping criteria
    def _update_weights(self, epoch, logs):
        """
        Updates the weights in the 'LinearPass' layer.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        logs : dict
            Contains information on the training results (accuracy and loss 
            values).

        Returns
        -------
        None.

        """
        
        layer = self.model.get_layer(self.layer_name)
        
        if (not self.trigger.criterion_features and
            self.trigger.d >= self.params.d_min):
            
            weights, loss = self._prune_weights(epoch) # pruned weights
            
            if np.array_equal(weights, self.log.weights[-1]):
                return None
            
            # this whole part is skipped if requirements for trigger condition
            # 2 are not fulfilled
            layer.set_weights([weights])  
            self.log.update(epoch, loss)
            self.trigger.update_trigger(logs)
        
            print(f"Epoch {epoch} - Weight Update:\n"
                  f"Pruned {int(len(weights) - np.sum(weights))} feature(s). "
                  f"Left with {self.log.n_features[-1]} feature(s).\n")
        
    
    def _update_stopping_criteria(self):
        """
        Updates the boolean stopping criteria values. If the number of desired
        features is attained, it sets the feature criterion True and if the
        algorithm surpassed the maximum number of training epochs without an 
        update, it will set the max criterion True.

        Returns
        -------
        None.

        """
        
        
        #stopping criterion 1
        if self.log.n_features[-1] <= self.n_features:
            self.trigger.criterion_features = True
        
        #stopping criterion 2
        if self.trigger.d_prune >= self.params.d_max:
            self.trigger.criterion_max = True
    
    def _get_n_features(self):
        """
        Calculates the number of nodes or features n_features at each 
        iteration i. The applied function depends on the pruning type and its 
        specific parameters (rate r_p, number n_p), the number of weights or 
        initial number of features n_in and the number of times n_i that the 
        features already have been pruned.
        
        Exponential:
            n_features = n_in * (1-r_p)**n_i
        
        Linear:
            n_features = n_in - n_p*n_i
        
        Returns
        -------
        n_features : int
            Number of features that are still active after the pruning.
        
        Raises
        ------
        NameError
            Raised if the FS pruning type is not valid and there is no 
            corresponding function for it.
        
        """
        if self.params.pruning_type == "exponential":
            n_features = int(self.n_in * (1 - self.params.pruning_rate) 
                             ** len(self.log.pruning_epochs))
            
            # if the same int is calculated twice, it will be subtracted by one
            if n_features == self.log.n_features[-1]:
                n_features -= 1
        
        elif self.params.pruning_type == "linear":
            n_features = int(self.n_in - (self.params.n_prune
                                          * len(self.log.pruning_epochs)))
        
        else:
            raise NameError(f"'{self.trigger_params.pruning_type}' is not "
                            "valid as 'pruning_type'. Try 'exponential' or "
                            "'linear' instead.")
        
        # securing minimum number of features
        if n_features <= self.n_features:
            n_features = self.n_features

        return n_features    

    def _get_indices(self, H_features, n_features):
        """
        Get the features with the most important information.

        Parameters
        ----------
        H_features : ndarray
            Loss matrix of the features.
        n_features : int
            Number of features that are to be kept.

        Returns
        -------
        index : ndarray
            Indices of the most important weights.

        """
        # the convenience factor is an adaptive factor that shall encourage 
        # the feature selection callback to trigger after the second stage 
        # pruning trigger is not pulled in the first few attempts 
        # if self.trigger.d >= self.params.d_min:
        #     convenience_factor = self.params.d_min / self.trigger.d
        # else:
        #     convenience_factor = 1.0
        
        if n_features == self.log.n_features[-1]:
            index = self.log.index[-1]
        
        elif self.params.loocv: # highest H_features shall endure
            prune = H_features[np.argsort(H_features)[:-n_features]]
            keep_index = np.argsort(H_features)[-n_features:]
            keep = H_features[keep_index]
            prune_max = np.amax(prune)
            keep_max = np.amax(keep)
            # the difference between kept and pruned features shall be at 
            # least 10 % of the maximum loss value 
            if (keep_max - prune_max) >= (self.params.d_loss_ratio 
                                          # * convenience_factor 
                                          * keep_max):
                index = self.log.index[-1][keep_index]
            else:
                index = self.log.index[-1]
                
        
        else: # lowest H_features shall endure
            prune = H_features[np.argsort(H_features)[n_features:]]
            keep_index = np.argsort(H_features)[:n_features]
            keep = H_features[keep_index]
            prune_max = np.amax(prune)
            prune_min = np.amin(prune)
            keep_min = np.amin(keep)
            # the difference between kept and pruned features shall be at 
            # least 10 % of the minimum loss value
            if (prune_min - keep_min) >= (self.params.d_loss_ratio 
                                          # * convenience_factor 
                                          * prune_max):
                index = self.log.index[-1][keep_index]
            else:
                index = self.log.index[-1]
                
        return index
    
    def _prune_weights(self, epoch):
        """
        Class method to calculate new weight matrix for the 'LinearPass' layer.

        Parameters
        ----------
        weights : float
            Previously used weight matrix. Initiallized with all weights being 
            1.
        n_min : int
            Minimum number of nodes to be leftover. If 'None', the weights are 
            updated until the threshold condition cannot be met anymore.

        Returns
        -------
        weights : ndarray
            Calculated new sparse weight matrix.
        H_features : ndarray
            The loss values for each feature.

        """        
        # get prediction data for the measure
        data, labels = self.map_validation_data()
        
        P = labels
        Q = self.model.predict(data)
        Q = np.where(Q > 1e-8, Q, 1e-8)
        
        # cross-entropy H measures uncertainty of possible outcomes and is the 
        # best suited loss metric for multiclass classification tasks
        H_samples = self.params.calculate_loss(P, Q) # loss for all samples
        
        # arranges sample losses according to their respective feature
        H_feature_samples = np.array(np.array_split(H_samples, 
                                                    self.log.n_features[-1]))
        
        # calculates mean of all entropies as loss
        H_features = np.mean(H_feature_samples, axis = 1)
        
        # calculate the amount of features after the pruning iteration
        n_features = self._get_n_features()
        
        # get indices that shall be kept
        index = self._get_indices(H_features, n_features)
        
        weights = np.zeros(self.log.weights[0].shape)
        weights[index] = 1
        
        return weights, H_features