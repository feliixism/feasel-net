import numpy as np
from ..metrics import cross_entropy, entropy

# mapping for the different functionalities of the feature selection callback
_METRIC_MAP = {'accuracy': {'thresh': 0.95,
                            'grad': 0.0},
               'loss': {'thresh': 0.0,
                        'grad': 1e-3},
               'val_accuracy': {'thresh': 0.95,
                                'grad': 0.0},
               'val_loss': {'thresh': 0.0,
                            'grad': 1e-3}}

_METRIC_TYPES = {'accuracy': ['accuracy', 'acc', 'acc.'],
                 'loss': ['loss'],
                 'val_accuracy': ['val_accuracy', 'val_acc'],
                 'val_loss': ['val_loss']}

_LOSS_MAP = {'cross_entropy': cross_entropy,
             'entropy': entropy}

_LOSS_TYPES = {'cross_entropy': ['cross_entropy', 'CE'],
               'entropy': ['entropy']}

_PRUNING_MAP = {'exponential': {'pruning_rate': 0.2,
                                'n_prune': None},
                'linear': {'pruning_rate': None,
                           'n_prune': 1}}

_PRUNING_TYPES = {'exponential': ['exponential', 'exp', 'exp.', 'ex', 'ex.'],
                  'linear': ['linear', 'lin', 'lin.']}

class CallbackParams:
    def __init__(self, 
                 metric="accuracy",
                 loss="cross_entropy", 
                 pruning_type="exponential",
                 thresh=None, 
                 grad=None,
                 d_min=20, 
                 d_max=500,
                 pruning_rate=None, 
                 n_prune=None,
                 n_samples=None,
                 d_loss_ratio=0.1,
                 loocv=True,
                 decay=0.00):
        """
        Parameter class for the trigger control of the leave-one-out 
        cross-validation (LOOCV) based feature selection callback.

        Parameters
        ----------
        metric : str, optional
            The metric that is monitored. See _METRIC_TYPES for the possible
            arguments. The type of metric determines the default arguments of 
            'thresh' and 'grad'. The default is "accuracy".
        loss : str, optional
            The loss function that is exectuted for the evaluation of the 
            features' importances. See _LOSS_TYPES for the possible arguments.
            The default is "cross_entropy".
        pruning_type : str, optional
            The type of node reduction for the recursive feature elimination. 
            See _PRUNING_TYPES for the possible arguments. The type of pruning 
            determines the default arguments of 'pruning_rate' and 'n_prune'. 
            The default is "exponential".
        thresh : float, optional
            The threshold value that has to be surpassed in order to trigger 
            the callback. If the metric 'accuracy' or 'val_accuracy' is chosen,
            it is set to 0.95 by default. The corresponding default value for
            the loss metrics is 0.1.
        grad : float, optional
            The gradient threshold value that has to be surpassed in order to 
            trigger the callback. The default is 0.001.
        d_min : int, optional
            The number of epochs in which the trigger conditions have to be 
            met consistently. The default is 20.
        d_max : int, optional
            The maximum number of epochs in which the optimizer tries to meet 
            the trigger conditions. If None, it will try to reach the 
            thresholds until the end of the actual training process set by the 
            number of epochs in the keras model.fit() method. 
            The default is None.
        pruning_rate : float, optional
            The percentage of nodes being eliminated after the update method is 
            triggered. The rate is only necessary if 'exponential' is chosen 
            as 'pruning_type' and it is set to None otherwise. 
            The default is 0.2.
        n_prune : int, optional
            The absolute number of nodes that are eliminated at each iteration. 
            The number is only necessary if 'linear' is chosen as 
            'pruning_type'. The default is None.
        n_samples : int, optional
            The number of validation samples that are chosen for the 
            evaluation of the most important features. If None, it will not 
            use any validation data, but only evaluates the weights in the 
            model without any sample data. This implies a significantly faster 
            evaluation method on the one hand and worse consistency in the
            feature selection results on the other hand. The default is None.
        d_loss_ratio : float, optional
            The minimum ratio of the difference of the first pruned and last 
            kept feature and the best overall feature loss that has to be 
            obtained in order to trigger pruning (2nd pruning stage). The 
            default is 0.1.
        loocv : bool, optional
            If True, it will apply the principle of the LOOCV and mask only
            one feature per classification. If False, it will use the inverse 
            LOOCV and mask every feature but one. The default is True.
        decay : float
            Sets an adaptive thresholding value. If set, it decreases the 
            threshold (thresh or grad respectively) by this value until it 
            meets trigger conditions. The threshold will be set to its original 
            state after the trigger is being pulled again. The deafult is None.
            
        Returns
        -------
        None.

        """
        self.metric = self._get_metric(metric)
        self.loss = self._get_loss(loss)
        self.pruning_type = self._get_pruning_type(pruning_type)
        self.thresh = self._get_thresh(thresh)
        self.grad = self._get_grad(grad)
        self.d_min = d_min
        self.d_max = d_max
        self.pruning_rate = self._get_pruning_rate(pruning_rate)
        self.n_prune = self._get_n_prune(n_prune)
        self.n_samples = n_samples
        self.d_loss_ratio = d_loss_ratio
        self.loocv = loocv
        self.decay = decay
    
    def __repr__(self):
        return ('Parmeter container for the Feature Selection Callback\n'
                f'{self.__dict__}')
    
    def _get_metric(self, metric):
        """
        Provides different aliases for all possible metrices and searches for
        the corresponding metric (e.g. 'acc' --> 'accuracy').

        Parameters
        ----------
        metric : str
            Metric alias.

        Raises
        ------
        NameError
            If metric is not a valid alias.

        Returns
        -------
        METRIC : str
            The proper metric used for the following operations inside the 
            class.

        """
        for METRIC in _METRIC_TYPES:
            if metric in _METRIC_TYPES[f'{METRIC}']:
                return METRIC
        raise NameError(f"'{metric}' is not a valid metric.")
        
    def _get_loss(self, loss):
        """
        Provides different aliases for all possible losses and searches for
        the corresponding loss (e.g. 'CE' --> 'cross_entropy').

        Parameters
        ----------
        loss : str
            Loss alias.

        Raises
        ------
        NameError
            If loss is not a valid alias.

        Returns
        -------
        LOSS : str
            The proper loss used for the following operations inside the class.

        """
        for LOSS in _LOSS_TYPES:
            if loss in _LOSS_TYPES[f'{LOSS}']:
                return LOSS
        raise NameError(f"'{loss}' is not a valid loss.")
    
    def _get_pruning_type(self, pruning_type):
        """
        Provides different aliases for all possible pruning types and searches 
        for the corresponding type (e.g. 'exp' --> 'exponential').

        Parameters
        ----------
        pruning_type : str
            Pruning type alias.

        Raises
        ------
        NameError
            If pruning type is not a valid alias.

        Returns
        -------
        PRUNING_TYPE : str
            The proper type used for the following operations inside the class.

        """
        for PRUNING_TYPE in _PRUNING_TYPES:
            if pruning_type in _PRUNING_TYPES[f'{PRUNING_TYPE}']:
                return PRUNING_TYPE
        raise NameError(f"'{pruning_type}' is not a valid pruning type.")
        
    def _get_thresh(self, thresh):
        """
        Looks up the metric map for the default threshold value if no other 
        value is given.

        Parameters
        ----------
        thresh : float
            Threshold value for the feature selection callback.

        Returns
        -------
        thresh : float
            Threshold value for the feature selection callback.

        """
        if not thresh or self.metric in ['loss', 'val_loss']:
            thresh = _METRIC_MAP[f'{self.metric}']['thresh']            
        return thresh
    
    def _get_grad(self, grad):
        """
        Looks up the metric map for the default gradient value if no other 
        value is given.

        Parameters
        ----------
        grad : float
            Gradient value for the feature selection callback.

        Returns
        -------
        grad : float
            Gradient value for the feature selection callback.

        """
        if not grad or self.metric in ['accuracy', 'val_accuracy']:
            grad = _METRIC_MAP[f'{self.metric}']['grad']
        return grad
    
    def _get_pruning_rate(self, pruning_rate):
        """
        Looks up the pruning map for the default pruning rate value if no other 
        value is given.

        Parameters
        ----------
        pruning_rate : float
            Pruning rate value for the feature selection callback.

        Returns
        -------
        pruning_rate : float
            Pruning rate value for the feature selection callback.

        """
        if not pruning_rate:
            pruning_rate = _PRUNING_MAP[f'{self.pruning_type}']['pruning_rate']
        return pruning_rate
    
    def _get_n_prune(self, n_prune):
        """
        Looks up the pruning map for the default pruning number if no other 
        value is given.

        Parameters
        ----------
        n_prune : float
            Pruning number for the feature selection callback.

        Returns
        -------
        n_prune : float
            Pruning number for the feature selection callback.

        """
        if not n_prune:
            n_prune = _PRUNING_MAP[f'{self.pruning_type}']['n_prune']
        return n_prune
    
    def get_map(self, type):
        """
        Returns the specified map with all possible options.

        Parameters
        ----------
        type : str
            Name of the map. Possible options are 'metric', 'loss' and 
            'pruning'.

        Raises
        ------
        NameError
            If wrong type is given.

        Returns
        -------
        map : dict
            The specific map dict.

        """
        if type == 'metric':
            return _METRIC_MAP
        
        elif type == 'loss':
            return _LOSS_MAP
        
        elif type == 'pruning':
            return _PRUNING_MAP
        
        else:
            raise NameError(f"Unable to find a map for '{type}' type (valid "
                            "types are: metric, loss and pruning.")
    
    def calculate_loss(self, P, Q):
        """
        Calculates the loss between predicted Q and actual P class.

        Parameters
        ----------
        P : ndarray
            The actual data.
        Q : ndarray
            The predicted data.

        Returns
        -------
        loss : ndarray
            The losses between each data set.

        """
        loss = _LOSS_MAP[f'{self.loss}'](P,Q)
        return loss
    
class CallbackLog:
    def __init__(self, fs_params, model, layer_name):
        """
        A callback log class that tracks all relevant and interesting data 
        during the callback such as the loss values for the decisoion whether 
        to keep features or not (class variable: 'loss').

        Parameters
        ----------
        fs_params : CallbackParams
            The parameter object for the feature selection algorithm. It 
            contains all relevant initializing parameters.
        model : keras.engine.training.Model
            The neural network architecture plus the pre-trained weights.
        layer_name : str
            The layer where the feature selection is applied.

        Returns
        -------
        None.

        """
        self.fs_params = fs_params
        self.model = model
        self.layer_name = layer_name
        self._check_layer()
        
    def __repr__(self):
        return ('Log container for the Feature Selection Callback\n'
                f'{self.__dict__}')
    
    def _check_layer(self):
        """
        Checks whether the layer is suited for the callback: must be a 
        'LinearPass' layer type.

        Raises
        ------
        NameError
            Raised if layer is not of type 'LinearPass'.

        Returns
        -------
        None.

        """
        if (self.model.get_layer(self.layer_name).__class__.__name__ 
            == "LinearPass"):
            self.layer = self.model.get_layer(self.layer_name)
        
        else:
            raise NameError(f"'{self.layer}' is not a 'LinearPass' layer. "
                            "Please choose another layer or make sure that the"
                            " specified layer is part of your model.")
    
    def initialize(self, logs):
        """
        Initializes the Log class with all the values after the very first 
        training epoch. This method has to be called after the class 
        instantiation because it needs the model information that, at the time
        of instantiation, has not been instantiated yet and thus cannot be
        referred to.
        
        Initializes the following parameters:
            - pruning_epochs: A list of all epochs when the callback has been 
                triggered.
            - index: A list of ndarray matrices with all the indexes where the
                signal has been masked or pruned. 
            - weights: A list of ndarray matrices with the actual signal masks.
                This is a very interesting class variable to inspect the 
                feature selection process, which features have been pruned 
                first and which survived longer.
            - n_features: A list with the number of features left after every
                iteration.
            - loss: A list of ndarray matrices with the loss values for every 
                feature after the evaluation with the validation set.
            
        Parameters
        ----------
        logs : dict, optional
            A dictionary of the current epoch's metric. The default is {}.

        Returns
        -------
        None.

        """
        mask = self.layer.weights[0].numpy()
        self.pruning_epochs = [0]
        self.index = [np.argwhere(mask == 1).squeeze()]
        self.weights = [np.array(mask)]
        self.n_features = [int(np.sum(mask))]
        self.loss = [np.ones(mask.shape) * logs[f'{self.fs_params.metric}']]
    
    def update(self, epoch, loss):
        """
        Updates all class variables initialized by initialize().

        Parameters
        ----------
        epoch : int
            The current training epoch.
        loss : ndarray
            The loss values for each feature.

        Returns
        -------
        None.

        """
        self.layer = self.model.get_layer(self.layer_name)
        mask = self.layer.weights[0].numpy()
        self.pruning_epochs.append(epoch)
        self.index.append(np.argwhere(mask == 1).squeeze())
        self.weights.append(np.array(mask))
        self.n_features.append(int(np.sum(mask)))
        self.loss.append(loss)

class CallbackTrigger(CallbackLog):
    def __init__(self, fs_params, model, layer):
        """
        A class that keeps track of the current state of all relevant trigger
        variables. The class variables decide, whether to prune features or 
        not. It is a subclass of the CallbackLog class since both need the 
        same information.

        Parameters
        ----------
        fs_params : CallbackParams
            The parameter object for the feature selection algorithm. It 
            contains all relevant initializing parameters.
        model : keras.engine.training.Model
            The neural network architecture plus the pre-trained weights.
        layer_name : str
            The layer where the feature selection is applied.

        Returns
        -------
        None.

        """
        super().__init__(fs_params, model, layer)
    
    def __repr__(self):
        return ('Trigger container for the Feature Selection Callback\n'
                f'{self.__dict__}')

    def initialize(self, logs):
        """
        Initializes the Log class with all the values after the very first 
        training epoch. This method has to be called after the class 
        instantiation because it needs the model information that, at the time
        of instantiation, has not been instantiated yet and thus cannot be
        referred to.

        Initializes the following parameters:
            - outcome: The current value of the spectated loss or 
                classification accuracy.
            - d: The number of epochs since the first time the first trigger 
                condition (surpassing the threshold or gradient value several 
                times in a row) has been met in a row. 
            - d_hit: The number of epochs since the last time the first trigger
                condition has been met.
            - d-prune: The number of epochs since the last time the feature
                selection algorithm was triggered.
            - gradient: The gradient of the loss value at the current epoch.
            - thresh: The threshold of accuracy or loss values that have to be 
                surpassed (acc > thresh and loss < thresh). It adaptively 
                changes over time, if the actual threshold cannot be reached.
            - grad: The gradient value that has to be surpassed. It adaptively 
                changes over time, if the actual threshold cannot be reached.
            - epochs: The current epoch of training.
            - criterion_max: The first stopping criterion (training too long 
                without reaching desired number of features).
            - criterion_features: The second stopping criterion (reaching 
                desired number of features).
            - first_prune: Is set True after the first pruning. If it is True,
                it will allow the adative thresholds and gradients.
            
        Parameters
        ----------
        logs : dict, optional
            A dictionary of the current epoch's metric. The default is {}.

        Returns
        -------
        None.

        """
        self._outcome = [logs[f'{self.fs_params.metric}']]
        self.outcome = self._outcome[-1]
        self.d = 0
        self.d_hit = 0
        self.d_prune = 0
        self.gradient = np.inf
        self.thresh = self.fs_params.thresh
        self.grad = self.fs_params.grad
        self.epochs = 0
        self.criterion_max = False
        self.criterion_features = False
        self.first_prune = False
        self.converged = False
    
    def update(self, logs):
        """
        Updates many class variables initialized by initialize() after every
        epoch.

        Parameters
        ----------
        loss : ndarray
            The loss values for each feature.

        Returns
        -------
        None.

        """
        self._outcome.append(logs[f'{self.fs_params.metric}'])
        self.outcome = self._outcome[-1]
        self.epochs += 1
        d_min = self.fs_params.d_min
        if len(self._outcome) > d_min:
            self.gradient = self._get_grad(logs, d_min)
            
    def update_miss(self, logs):
        """
        Updates all other class variables initialized by initialize() after 
        every miss (not meeting the first criterion).

        Parameters
        ----------
        loss : ndarray
            The loss values for each feature.

        Returns
        -------
        None.
        
        """
        self.update(logs)
        self.d = 0
        self.d_hit += 1
        self.d_prune += 1
        self.epochs += 1
        
        # adaptive thresholding
        d_miss_after_min = self.d_hit - self.fs_params.d_min
        if d_miss_after_min > 0 and self.first_prune:
            
            if self.fs_params.metric in ['accuracy', 'val_accuracy']:
                self.thresh = self.fs_params.thresh - (d_miss_after_min 
                                                       * self.fs_params.decay)
            else:
                self.thresh = self.fs_params.thresh + (d_miss_after_min 
                                                       * self.fs_params.decay)
            
            self.grad = self.fs_params.grad + (d_miss_after_min 
                                               * self.fs_params.grad)
        
    def update_hit(self, logs):
        """
        Updates all other class variables initialized by initialize() after 
        every hit (meeting the first criterion).

        Parameters
        ----------
        loss : ndarray
            The loss values for each feature.

        Returns
        -------
        None.
        
        """
        self.update(logs)
        self.d += 1
        self.d_hit = 0
        self.d_prune += 1
        self.grad = self.fs_params.grad
        self.epochs += 1
        
    def update_trigger(self, logs):
        """
        Updates all other class variables initialized by initialize() after 
        every time the feature callback is triggered (meeting first and second
        criterion).

        Parameters
        ----------
        loss : ndarray
            The loss values for each feature.

        Returns
        -------
        None.
        
        """
        self.update(logs)
        self.d = 0
        self.d_hit = 0
        self.d_prune = 0
        self.thresh = self.fs_params.thresh
        self.first_prune = True
        self.epochs += 1
        
    def _get_grad(self, logs, s):
        """
        Calculates the gradient at the current epoch.

        Parameters
        ----------
        logs : dict, optional
            A dictionary of the current epoch's metric. The default is {}.
        s : int
            The number of epochs that are considered to calculate the 
            gradient.

        Returns
        -------
        grad : float
            The gradient of the loss.

        """
        grad = np.abs((self._outcome[-s-1]-self._outcome[-1]) / s)
        return grad