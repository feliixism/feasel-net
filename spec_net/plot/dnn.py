import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tensorflow.keras.utils import plot_model
import matplotlib.ticker as mtick
from pathlib import Path

from .utils import _default

class DNN:
    def __init__(self, model_container):
        self.model_container = model_container
        self.default = _default.AdvancedPlot()
        self.default.set_default()
        self.default.enable_tex()
        self._model = self.model_container.model
        self._directory = self.set_directory()
        
    def _get_layer_positions(self):
        layer_dict = {}
        for i in range(len(self._model.layers)):
            layer_dict[f"{self.layers[i].name}"] = i
        return layer_dict
    
    def _get_layer_name(self):
        layer_names = []
        for i in range(len(self._model.layers)):
            layer_names.append(self._model.layers[i].name)
        return layer_names
    
    def _get_layer_information(self, layer_names, information_type):
        layer_values = []
        names = []
        for i in range(len(layer_names)):
            try:
                weights, bias = self.model_container.get_weights(layer_names[i])
                names.append(layer_names[i])
                if information_type == "bias":
                    layer_values.append(bias)
                elif information_type == "weights":
                    layer_values.append(weights)
            except:
                pass
        return layer_values, names
    
    def set_directory(self, directory=None):
        if not directory:
            directory = (Path.cwd() / "plots")
        else:
            directory = Path(directory)
        return directory
    
    def set_modelname(self, modelname=None):
        if not modelname:
            modelname = f"{self.model_container.name}"
        else:
            modelname = modelname
        return modelname
            
    def set_path(self, directory=None, modelname=None):
        path = self.set_directory(directory) / self.set_modelname(modelname)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def model(self, directory=None, modelname=None):
        # The option "layer_range=None" has not been implemented in keras yet.
        path = self.set_path(directory, modelname)
        filename = path / Path("model" + self.model_container.time + ".png")
        plot_model(self._model, 
                   filename, 
                   show_shapes=True, 
                   dpi=100)
        
    def history(self, *metrics, **kwargs):
        history = self.model_container.history.history
        n_data = len(metrics)
        if len(metrics) == 0:
            metrics = list(history.keys())
            n_plots = 0
            for i in range(len(metrics)):
                if "val" in metrics[i]:
                    continue
                n_plots += 1
        else:
            metrics = np.array(metrics)
            n_plots = n_data
        
        # definition of subplot size
        hor = np.ceil(n_plots / 4).astype(int)
        if n_plots >= 4:
            vert = 4
        else: 
            vert = n_plots
        
        # plotting figure    
        fig = plt.figure()
        fig.canvas.set_window_title("Losses and Metrics")
        
        n = 1
        for i in range(len(metrics)):
            if "val" in metrics[i]:
                continue
            
            ax = fig.add_subplot(vert, hor, n)
            plt.sca(ax)
            
            # scaling y-axis
            max_value = max(history[f"{metrics[i]}"])
            try:
                max_val_value = max(history[f"val_{metrics[i]}"])
                plt.ylim(0, max(max_value, max_val_value))
            except:
                plt.ylim(0, max_value)
            
            if "acc" in metrics[i]:
                plt.ylim(0, 1)
                plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
                plt.ylabel(r"Accuracy [$\%$]")
            
            elif "loss" in metrics[i]:
                plt.ylabel("Loss [-]")
            
            elif "mse" in metrics[i]:
                plt.ylabel("MSE [-]")
            
            # plt.title(f"{metrics[i]}".replace("_", " "))
            plt.plot(history[f"{metrics[i]}"], label = "Training")
            try:
                plt.plot(history[f"val_{metrics[i]}"], label = "Validation")
            except:
                pass
            
            # scaling x-axis
            plt.xlim(0, len(history[f"{metrics[i]}"]))
            plt.xticks(np.arange(0, len(history[f"{metrics[i]}"]) + 1, len(history[f"{metrics[i]}"]) / 5).astype(int))
            plt.grid(True)
            if n == 1:
                plt.legend()
            # if n == len(metrics):
            #     plt.xticks(np.arange(0, len(history[f"{metrics[i]}"]) + 1, len(history[f"{metrics[i]}"]) / 5).astype(int))
            # else:
            #     plt.xticks(np.arange(0, len(history[f"{metrics[i]}"]) + 1, len(history[f"{metrics[i]}"]) / 5), [])
            
            # x-label only on lowest plots in each column
            for row in range(hor):
                if n == n_plots - row:
                    plt.xlabel("Epochs [-]")
    
            n += 1
            
        fig.tight_layout() 
        fig.show()
    
    def _aspect_ratio(n_plots, max_vert = 4):
        
        hor = np.ceil(n_plots / max_vert).astype(int)
        vert = np.ceil(n_plots / hor)
        
        return vert, hor
    
    def bias(self, *layer_names):
        """
        Plots bias values.
        
        Parameters
        ----------
        layer_names : str (or: list of str)
            Defines name of layer(s).
        """
        if len(layer_names) == 0:
            layer_names = self._get_layer_name(self)
        
        biases, names = self._get_layer_information(self, layer_names, information_type = "bias")
    
        n_biases = len(biases)
        
        vert, hor = self._aspect_ratio(n_biases)
        
        fig = plt.figure()
        fig.canvas.set_window_title("Biases")
        for i in range(n_biases - 1):
            i += 1
            ax = fig.add_subplot(vert, hor, i)
            plt.sca(ax)
            x = np.round(np.arange(0, biases[i].shape[0]), 0)
            plt.plot(x, biases[i], label = f"{names[i]}")
            plt.xticks(np.arange(0, len(biases[i]) + 1, len(biases[i]) / 5).astype(int))
            plt.xlim(0, len(biases[i]))
            plt.figaspect(fmt)
            plt.legend(loc = 4)
            plt.grid(True)
        fig.show()
    
    def _batch_norm(self, layers):
        for layer in layers:
            current_layer = self._model.layers[layer]
            gamma = current_layer._trainable_weights[0].numpy()
            beta = current_layer._trainable_weights[1].numpy()
            
            mean = current_layer._non_trainable_weights[0].numpy()
            variance = current_layer._non_trainable_weights[1].numpy()
            stdev = np.sqrt(variance)
            
            upper = mean + stdev
            lower = mean - stdev
            
            fig = plt.figure()
            fig.canvas.set_window_title(f"Parameters of BatchNorm-Layer: {current_layer.name}")
            fig.clf()
       
            ax = fig.add_subplot(1, 1, 1)
            plt.sca(ax)
            
            x = np.arange(len(gamma))
            plt.plot(gamma, label = "$\\gamma$")
            plt.plot(beta, label = "$\\beta$")
            plt.plot(mean, label = "$\\bar{x}$", color = "k")
            plt.plot(upper, color = "gray", alpha = 0.2)
            plt.plot(lower, color = "gray", alpha = 0.2)
            plt.fill_between(x, upper, lower, color = "gray", alpha = 0.2)
            plt.grid(True)
            plt.legend(loc = "lower center", ncol = 3)
            plt.xlim(0, len(x) - 1)
            plt.xlabel("Nodes [-]")
                
            fig.tight_layout()
    
    def _dense(self, layers):
        for layer in layers:
            current_layer = self._model.layers[layer]
            weights = current_layer._trainable_weights[0].numpy()
            
            fig = plt.figure()
            fig.canvas.set_window_title(f"First Four Node Parameters of Dense-Layer: {current_layer.name}")
            fig.clf()
       
            vert, hor = self._aspect_ratio(4)
            
            node = 0
            for node in range(4):
                node += 1
                ax = fig.add_subplot(vert, hor, node)
                plt.sca(ax)
                plt.title(f"Node {node}")            
                plt.plot(weights[node])
                plt.grid(True)
                plt.xlim(0, len(weights[node]) - 1)
                plt.xlabel("Nodes [-]")
                
            fig.tight_layout()
        
    def _conv1d(self, layers):
        for layer in layers:
            current_layer = self._model.layers[layer]
            conv_weights = current_layer._trainable_weights[0].numpy()
            fig = plt.figure()
            fig.canvas.set_window_title(f"Parameters of Conv-Layer: {current_layer.name}")
            fig.clf()
        
            vert, hor = self._aspect_ratio(conv_weights.shape[-1])
        
            kernel = 0
            for kernel in range(conv_weights.shape[-1]):
                kernel += 1
                ax = fig.add_subplot(vert, hor, kernel)
                plt.sca(ax)
                plt.title(f"Kernel {kernel}")
                for n_FeatureMap in range(conv_weights.shape[1]):
                    plt.plot(conv_weights[:, n_FeatureMap, kernel - 1])
                plt.grid(True)
                plt.xlim(0, len(conv_weights) - 1)
                plt.xlabel("Nodes [-]")
            fig.tight_layout()
    
    def has_callback(self):
        if not hasattr(self.model_container, "callback"):
            raise NameError("Could not find the callback object. "
                            "Please ensure that it is instantiated correctly.")
        elif len(self.model_container.callback.log.weights) == 1:
            raise NotImplementedError	("The feature selection did not return "
                                       "anything printworthy.")
    
    def pruning_history(self):
        fig = plt.figure('Pruning History')
        plt.clf()
        ax = fig.add_subplot(111)
        ax = self._pruning_history(ax)
        plt.tight_layout()
    
    def _pruning_history(self, ax):
        
        self.has_callback() # ensure that there is a callback to work with
        
        # plot content and informative variables
        log = self.model_container.callback.log
        
        weights = np.array(log.weights)
        n_kills, n_features = weights.shape
        n_features = np.array(log.n_features).astype(int)
        epochs = np.array(log.pruning_epochs)
        
        # plotting settings for the pruning history
        ax.plot(epochs, n_features, color = "k", marker = ".")
        ax.set_xlabel("Pruning Epochs")
        ax.set_ylabel("Number of Features")
        
        if epochs[1] != epochs[-2]:
            # first prune
            ax.axvline(epochs[1], color = "k", ls = "-.")
            ax.text(epochs[1], n_features[0] / 2, 
                    f"first prune: {epochs[1]}", 
                    rotation = "vertical", 
                    horizontalalignment = "right", 
                    verticalalignment = "center", color = "k")
            
            # last prune
            ax.axvline(epochs[-2], color = "k", ls = "-.")
            ax.text(epochs[-2], n_features[0] / 2, 
                    f"last prune: {epochs[-2]}", 
                    rotation = "vertical", 
                    horizontalalignment = "right", 
                    verticalalignment = "center", color = "k")
            
        else:
            # first prune
            ax.axvline(epochs[1], color = "k", ls = "-.")
            ax.text(epochs[1], n_features[0] / 2, 
                    f"only prune: {epochs[1]}", 
                    rotation = "vertical", 
                    horizontalalignment = "right", 
                    verticalalignment = "center", color = "k", size="x-small") 
        
        ax.set_ylim(0, n_features[0])
        ax.set_xlim(0, epochs[-1])
        
        ax.grid(True)
        
        return ax
    
    def mask_history(self, highlight=False):
        fig = plt.figure('Mask History')
        plt.clf()
        gs = fig.add_gridspec(1, 2, figure=fig, 
                              width_ratios=[1, 20],
                              wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 1])
        self._mask_history(ax1, highlight)
        ax2 = fig.add_subplot(gs[0, 0])
        self._colorbar(fig, ax2)
        fig.tight_layout()
        
    def _mask_history(self, ax, highlight):
        self.has_callback() # ensure that there is a callback to work with
        cmap1 = self.default.im_cmap
        # cmap1 = cmap1.reversed() <--depends on cmap (start w/ lighter colors)
        cmap2 = plt.get_cmap("Wistia")
        c = cmap2(255) # orange color of Wistia cmap
        
        # plot content and informative variables
        log = self.model_container.callback.log
        weights = np.array(log.weights)
        n_kills, n_features = weights.shape
        
        # plotting settings for the mask history
        masks = np.array(np.sum(weights, axis = 0), ndmin = 2)
        last = np.array(weights[-1], ndmin=2)
        
        if self.model_container.features is not None:
            features = self.model_container.features
            
            if self._check_linearity(features):
                ax.pcolormesh(features, np.arange(2), masks, cmap=cmap1)
                
                # single pcolormeshes for the plot of the remaining features
                if highlight:
                    ax.bar(features + 0.5, 
                           weights[-1], 
                           width=1, 
                           color=c)
                ax.set_xlim(features[0], features[-1])
                
                # inversion of the x-axis if required
                if features[-1] < features[0]:
                    ax.invert_xaxis()
            
            else:
                ax.imshow(masks, cmap = cmap1, aspect = "auto")
                # plot of the remaining features
                if highlight:
                    ax.imshow(last, cmap=cmap2, alpha=last, aspect = "auto")
                features = self.default.replace_char(features, '_', ' ')
            
                # enables 10 ticks in x-axis with string or  
                ratio = int(len(features) / 10)
                ax.set_xticks(np.arange(len(features))[::ratio])
                ax.set_xticklabels(features[::ratio], rotation=45, ha='right')
        
        else:
            ax.imshow(masks, cmap = cmap1, aspect = "auto")
            # plot of the remaining features
            if highlight:
                ax.imshow(last, cmap=cmap2, alpha=last, aspect = "auto")
        
        ax.set_xlabel("Features")
        ax.grid(False)
        ax.set_yticks([])
        return ax

    def _check_linearity(self, array):
        """
        Checks the linearity of an array, to ensure correct xtick labels.

        Parameters
        ----------
        array : nd-array
            The xtick-array.

        Returns
        -------
        bool
            True if xticks array is linear.

        """
        try:
            comparison_array = np.linspace(array[0], array[-1], len(array))
            return np.array_equal(array, comparison_array)
        except:
            return False
                    
    
    def _colorbar(self, fig, ax):
        self.has_callback() # ensure that there is a callback to work with
        cmap = self.default.im_cmap#.reversed()
        
        # plot content and informative variables
        log = self.model_container.callback.log
        weights = np.array(log.weights)
        n_kills, n_features = weights.shape
        n_nodes = np.array(log.n_features).astype(int)
        ratio = int(len(n_nodes) / 6) # secure 6 elements
        if ratio:
            n_nodes = n_nodes[::ratio]
        
        ticks = np.linspace(0, 1, len(n_nodes), endpoint=True)
        colorbar = fig.colorbar(cm.ScalarMappable(cmap = cmap), 
                                ticks = ticks, cax = ax, aspect = 20)
        colorbar.ax.set_yticklabels(n_nodes)
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.invert_yaxis()
        colorbar.set_label("Mask [I/O]")
        
        return colorbar
    
    def input_reduction(self, plot='both', highlight=False):
        
        self.has_callback() # ensure that there is a callback to work with
        
        fig = plt.figure("Input Reduction")
        plt.clf()
        
        if plot == "both":
            gs = fig.add_gridspec(2, 2, width_ratios = [1, 20], wspace = 0.05)
            ax1 = fig.add_subplot(gs[0, :])
            ax1 = self._pruning_history(ax1)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2 = self._colorbar(fig, ax2)
            ax3 = fig.add_subplot(gs[1, 1])
            ax3 = self._mask_history(ax3, highlight)
            
        elif plot == 'mask':
            plt.close()
            self.mask_history()
            fig = plt.gcf()
            fig.canvas.set_window_title("Input Reduction")
        
        elif plot == 'prune':
            plt.close()
            self.pruning_history()
            fig = plt.gcf()
            fig.canvas.set_window_title("Input Reduction")
        
        else:
            raise NameError(f"'{plot}' is not a valid argument for 'plot'. "
                            "Try 'mask', 'prune' or 'both' instead.")
        fig.tight_layout()
                   
    def weights(self, *layer_names):
        """
        Plots weights values.
        
        Parameters
        ----------
        layer_names : str (or: list of str)
            Defines name of layer(s).
        """
        
        executed = False
        if len(layer_names) == 0:
            layer_names = self._get_layer_name()
        
        weights, names = self._get_layer_information(layer_names = layer_names, information_type = "weights")
            
        n_weights = len(weights)
        positions = {"BatchNorm": [], "Dense": [], "Conv1D": [], "QuantizedDense": []}
        for i in range(n_weights):
            for pos in range(len(self._model.layers)):
                if self._model.layers[pos].name == names[i]:
                    break
            layer_type = self._model.layers[pos].__class__.__name__
            if layer_type == "BatchNormalization":
                positions["BatchNorm"].append(pos)
            elif layer_type == "Dense":
                positions["Dense"].append(pos)
            elif layer_type == "Conv1D":
                positions["Conv1D"].append(pos)
            elif layer_type == "QuantizedDense":
                positions["QuantizedDense"].append(pos)
        
        if len(positions["BatchNorm"]) != 0:
            self._batch_norm(positions["BatchNorm"])
            executed = True
        if len(positions["Dense"]) != 0:
            self._dense(positions["Dense"])
            executed = True
        if len(positions["Conv1D"]) != 0:
            self._conv1d(positions["Conv1D"])
            executed = True
        if len(positions["QuantizedDense"]) != 0:
            self._dense(positions["QuantizedDense"])
            executed = True
        if executed == False:
            print("Could not plot anything.\nTry one of these layers to plot: \n",
                  self._get_layer_positions())
        plt.tight_layout(0.1)
    
    def feature_maps(self, x, layer_name, lower_boundary = 4000, upper_boundary = 400):
        """
        Plots feature maps for given input x.
        
        Parameters
        ----------
        layer_name : str
            Defines name of layer.
    
        """
        plt.figure("Masked Input")
        test_model = self.model_container(inputs = self._model.inputs, outputs = self._model.get_layer(layer_name).output)
        sample_points = test_model.layers[0].input_shape[1]
        x_ = np.linspace(lower_boundary, upper_boundary, sample_points)
        # except:
        #     raise NameError(f"Layer '{layer_name}' is not a part of self.model_container '{self.model_container}'")
        feature_maps = self.model_container.test_model(x, model = test_model)
        
        n_feature_maps = len(feature_maps[0])
        
        # if kwargs["n_hor_plots"] is None:
        #     n_vert_plots = kwargs["n_vert_plots"]
        #     n_hor_plots = int(np.ceil(n_feature_maps / n_vert_plots))
        # else:
        #     n_hor_plots = kwargs["n_hor_plots"]
        #     n_vert_plots = int(np.ceil(n_feature_maps / n_hor_plots))
        # if len(feature_maps) >= 3:
        #     i = 1
        #     for _ in range(n_vert_plots):
        #         for _ in range(n_hor_plots):
        #             try:
        #                 ax = plt.subplot(n_vert_plots, n_hor_plots, i)
        #                 ax.set_xticks([])
        #                 ax.set_yticks([])
        #                 plt.plot(feature_maps[0, ..., i - 1])
        #             except:
        #                 pass
        #             i += 1
        # else:
        #     plt.plot(feature_maps[0])
        plt.plot(x_, x[0])
        plt.plot(x_, feature_maps[0])
        plt.xlabel("Wavenumbers [cm$^-1$]")
        # plt.ylabel("Rel. Scattering Intensity")
        plt.xlim(x_[0], x_[-1])
        plt.grid(True)
        return feature_maps
    
    def mask_generation(self, x):
        fig = plt.figure()
        fig.canvas.set_window_title("Mask Generation") 
        generator_layers = []
        for i in self.model_container.layer_names:
            if "generator" in i:
                generator_layers.append(i)
        for i in range(len(generator_layers)):
            test_model = self.model_container(inputs = self._model.inputs, outputs = self.model_container.get_layer(generator_layers[i]).output)
            feature_maps = self.model_container.test_model(x, model = test_model)
            ax = plt.subplot(len(generator_layers), 1, i + 1)
            plt.plot(feature_maps[0])
            plt.grid()
        fig.tight_layout()
        fig.show()
            
    
    # analyzers   
    def check_prediction(self, x, y, show_plot = False):
        x = self.model_container._convert_data(x, y)
        prediction = self._model.predict(x) * 100
        labels = self.model_container.labels.reshape([1, len(self.labels)])
        prediction = np.concatenate((labels, prediction), axis = 0)
        prediction = dict(zip(prediction[0], prediction[1].astype(float)))
        if show_plot == True:
            plt.figure(f"Bar plot of the prediction")
            plt.bar(range(len(prediction)), list(prediction.values()), align = "center")
            plt.xticks(range(len(prediction)), list(prediction.keys()), rotation = 90)
            plt.ylim(0, 100)
            plt.xlim(-0.5, len(prediction) - 0.5)
            plt.ylabel("Probability [%]")
            plt.grid(True)
        return prediction
    
    def confusion_matrix(self, y_pred = None, show_plot = False):
        from sklearn.metrics import confusion_matrix
        
        if y_pred is None:
                y_pred = self.model_container.test_model(self.x_train, self.model_container.y_train)
    
        y = self.model_container.y_test
        pred = np.argmax(y_pred[0], axis = 1)
        confusion_matrix = confusion_matrix(y, pred, normalize = "true")
        if show_plot == True:
            plt.figure("Confusion Matrix")
            plt.imshow(confusion_matrix)
            plt.xticks(range(self.n_labels), self.model_container.labels, rotation = 45, horizontalalignment = "right")
            plt.yticks(range(self.n_labels), self.model_container.labels)
            plt.tight_layout()
    
    def label_accuracy(self, y_pred = None):
        if y_pred is None:
            try:
                y_pred = self.model_container.test_model(self.x_train, self.model_container.y_train, model = self.model)
            except:
                y_pred = self.model_container.test_model(self.x_train, self.model_container.y_train)
        y = self.model_container.y_test
        label_accuracy = np.empty([self.model_container.n_labels])
        for i in range(self.n_labels):
            value = count = 0
            for j in range(len(y)):
                if y[j] == i:
                    value += y_pred[0][j, i]
                    count += 1
            label_accuracy[i] = np.round(value / count * 100, 2)
        mean = np.round(np.mean(label_accuracy), 2)
        plt.figure("Confidence of Prediction")
        plt.clf()
        plt.grid(True)
        plt.plot([-0.5, self.model_container.n_labels - 0.5], [mean, mean], color = "k", linestyle = "dashed")
        plt.text(self.n_labels - 0.5, mean + 0.002, f"overall accuracy: {mean}", ha = "right", va = "bottom", color = "k")
        plt.bar(range(self.n_labels), label_accuracy, color = "steelblue")
        plt.ylim(np.amin(label_accuracy) - 5, np.amax(label_accuracy) + 5)
        plt.xlim(-0.5, self.model_container.n_labels - 0.5)
        plt.xticks(range(self.n_labels), self.model_container.labels, rotation = 45, horizontalalignment = "right")
        plt.ylabel("accuracy [%]")
        for i in range(self.n_labels):
            plt.text(i, label_accuracy[i] + 0.25, label_accuracy[i], color = "steelblue", rotation = "vertical", va = "bottom", horizontalalignment = "center")
        plt.tight_layout()    
    
    def output(self, layername = None):
        i = np.random.randint(0, len(self.x_train[0]))
        plt.figure(f"Predicted Output {i}")
        array = np.array(self.x_train[i], ndmin = 2)
        feature_map = self._model.predict(array)
        pred = feature_map[0]
        try:
            plt.plot([0, self.model_container.y_train.shape[1]], [self.model_container.threshold, self.model_container.threshold])
            # plt.text(0, self.model_container.threshold, f"Threshold: {self.model_container.threshold}", horizontalalignment = "left", verticalalignment = "bottom")
            # plt.text(0, self.model_container.threshold, f"Sum: {np.sum(pred)}", horizontalalignment = "left", verticalalignment = "top")        
        except:
            pass
        plt.plot(pred)
        plt.grid(True)
        plt.show()
        
    def mask(self, y_max = None):
        if self.model_type == "spectral_net":
            random_pos = np.random.randint(0, len(self.x_train))
            
            label = self.model_container.labels[self.model_container.y_train[random_pos]]
            
            if self.model_container.classifier.architecture_type == "ann":
                signal = np.array(self.x_train[random_pos], ndmin = 2)
            else:
                signal = np.array(self.x_train[random_pos], ndmin = 3)
            
            mask = np.array(self.generator.x_train[random_pos], ndmin = 2)
            
            try:
                pred_model = self.model_container(inputs = self._model.inputs, outputs = self._model.get_layer('MaskedSignal').output)
                FM_signal = pred_model.predict(x = [signal, mask])
            except:
                raise NameError(f"Layer 'MaskedSignal' is not a part of self.model_container '{self.model_container.model_type}'")
            
            try:
                pred_model = self.model_container(inputs = self._model.inputs, outputs = self._model.get_layer("Generator").output)
                FM_mask = pred_model.predict(x = [signal, mask])
            except:
                raise NameError(f"Layer 'Generator' is not a part of self.model_container '{self.model_container.model_type}'")
            
            if self.model_container.classifier.architecture_type == "ann":
                masked_signal = FM_signal[0]
                mask = FM_mask[0]
                signal = signal[0]
            else:
                masked_signal = FM_signal[0][:, 0]
                signal = signal[0][:, 0]
                mask = FM_mask[0]
            
            x = np.round(np.linspace(3002, 401, 3600), 0)
            #for raman:
            masked_signal = np.flip(masked_signal)
            signal = np.flip(signal)
            mask = np.flip(mask)
            
            
            plt.figure(f"Masked Signal of {label}")
            
            plt.figtext(0.62, 0.9, "Fingerprint Region", horizontalalignment='center', color = "r")
            
            plt.subplot(2,1,1)
            plt.plot(x, masked_signal, label = "MaskedSignal")
            plt.plot(x, signal, label = "signal")
            plt.plot([1800, 1800], [0, 100], color = "r")
            plt.plot([900, 900], [0, 100], color = "r")
            if y_max is None:
                plt.ylim(0, 100)
            else:
                plt.ylim(0, y_max)
            plt.ylabel("Rel. Scattering Intensity [-]")
            plt.xlim(x[0], x[-1])
            plt.legend(loc = "lower left")
            plt.grid(True)
            
            plt.subplot(2,1,2)
            plt.plot(x, mask, label = "Mask")
            plt.plot([x[0], x[-1]], [self.model_container.generator.threshold, self.model_container.generator.threshold], color = "k")
            plt.plot([1800, 1800], [0, 1], color = "r")
            plt.plot([900, 900], [0, 1], color = "r")
            
            plt.ylim(0, 1.01)
            plt.xlim(x[0], x[-1])
            plt.xlabel("Wavenumber [cm$^{-1}$]")
            plt.legend(loc = "lower left")
            plt.grid(True)
        
        elif self.model_container.model_type == "mask_generator":
            i = np.random.randint(0, len(self.x_train[0]))
            plt.figure(f"Predicted Mask {i}")
            mask = np.array(self.x_train[i], ndmin = 2)
            feature_map = self._model.predict(mask)
            pred = feature_map[0]
            plt.plot([0, self.model_container.y_train.shape[1]], [self.model_container.threshold, self.model_container.threshold])
            plt.plot(pred)
            plt.text(10, self.model_container.threshold + 0.01, f"Threshold: {self.model_container.threshold}")
            plt.text(10, self.model_container.threshold - 0.05, f"Sum: {np.sum(pred)}")        
            plt.ylim(0, 1)
            plt.xlim(0, self.model_container.y_train.shape[1])
            plt.grid(True)
            plt.show()
        else:
            print("No mask available.")