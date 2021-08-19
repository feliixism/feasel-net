import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from keras.utils import plot_model
from keras import Model
import matplotlib.ticker as mtick

# default settings
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams.keys()
fmt = 16 / 9
from matplotlib import rc
rc('font', **{'family': "DejaVu Sans", 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)
color = [0., 116/255, 161/255]

class Base:
    def __init__(self, model_container):
        self.model_container = model_container
        
    def _get_layer_positions(self):
        layer_dict = {}
        for i in range(len(self.model_container.model.layers)):
            layer_dict[f"{self.model_container.model.layers[i].name}"] = i
        return layer_dict
    
    def _get_layer_name(self):
        layer_names = []
        for i in range(len(self.model_container.model.layers)):
            layer_names.append(self.model.layers[i].name)
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
    
    def model(self, filename = "self.model_container.png"):
        plot_model(self.model, filename, show_shapes = True)
        
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
            current_layer = self.model_container.model.layers[layer]
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
            fig.show()
    
    def _dense(self, layers):
        for layer in layers:
            current_layer = self.model_container.model.layers[layer]
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
            fig.show()
        
    def _conv1d(self, layers):
        for layer in layers:
            current_layer = self.model_container.model.layers[layer]
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
    
    # def input_reduction(self, plot = False, x_label = None):
    #     map = "Greys"
    #     cmap = cm.get_cmap(map)
        
    #     if not hasattr(self.model_container, "callback"):
    #         return
               
    #     else:
    #         callback = self.model_container.callback[0]
    #         weights = np.array(callback._log["weights"])
    #         n_kills, n_features = weights.shape
    #         n_nodes = np.array(callback._log["n_features"]).astype(int)
    #         epochs = np.array(callback._log["kill_epochs"])
            
    #         if plot:
    #             fig, axs = plt.subplots(2, 2)
    #             plt.clf()
    #             gs = axs[0, 0].get_gridspec()
    #             for ax in axs[1, :]:
    #                 ax.remove()
    #             ax_plot = fig.add_subplot(gs[1, :])
    #             ax_plot.plot(epochs, n_nodes, color = "k", marker = ".")
    #             ax_plot.set_xlabel("Killing Epochs [-]")
    #             ax_plot.set_ylabel("Number of Nodes [-]")
    #             ax_plot.axvline(epochs[1], color = "k", ls = "-.")
    #             ax_plot.text(epochs[1], n_nodes[0] / 2, f"first kill at epoch {epochs[1]}", 
    #                          rotation = "vertical", horizontalalignment = "right", 
    #                          verticalalignment = "center", color = "k")
    #             ax_plot.axvline(epochs[-1], color = "k", ls = "-.")
    #             ax_plot.text(epochs[-1], n_nodes[0] / 2, f"last kill at epoch {epochs[-1]}", 
    #                          rotation = "vertical", horizontalalignment = "right", 
    #                          verticalalignment = "center", color = "k")
    #             ax_plot.set_ylim(0, n_nodes[0])
    #             ax_plot.set_xlim(0, epochs[-1] + callback._log["delta_kill"][-1])
    #             ax_plot.grid(True)
    #         else:
    #             fig, axs = plt.subplots(1, 2)
    #             plt.clf()
                
    #         if not x_label:
    #             axs[0, 0].set_xlabel("Features [-]")
    #         else:
    #             axs[0, 0].set_xlabel(f"{x_label}")
                
    #         if hasattr(self, "features"):
    #             features = self.features
    #             axs[0, 0].set_xlabels(features)
            
    #         img = np.empty([n_nodes[0] + 1, n_nodes[0]])
    #         for i in range(n_nodes[0] + 1):
    #             img[i] = np.array(np.sum(weights, axis = 0), ndmin = 2)
    #         axs[0, 0].imshow(img, cmap = cmap, aspect = "auto")
    #         axs[0, 0].set_yticks([])
            
    #         axs[0, 1].colorbar(img)
            
    #         fig.set_window_title("Feature Selection")
    #         fig.tight_layout()
            
    #         # # colorbar
    #         # ticks = np.linspace(0, 1, len(n_nodes), endpoint=True)
    #         # cax = fig.add_axes([ax1.get_position().x1 + 0.02, ax1.get_position().y0, 0.02, ax1.get_position().y1 - ax1.get_position().y0])
    #         # cb = fig.colorbar(cm.ScalarMappable(cmap = map), ticks = ticks, cax = cax)
    #         # cb.ax.set_yticklabels(n_nodes)
    #         # cb.ax.invert_yaxis()
    #         # cb.set_label("Mask [I/O]")

    def input_reduction(self, plot = False, x_label = None, x_lim = None):
        map = "Greys"
        cmap = cm.get_cmap(map)
        
        if not hasattr(self.model_container, "callback"):
            return
               
        else:
            callback = self.model_container.callback[0]
            weights = np.array(callback._log["weights"])
            n_kills, n_features = weights.shape
            n_nodes = np.array(callback._log["n_features"]).astype(int)
            epochs = np.array(callback._log["kill_epochs"])
            
            fig = plt.figure()
            plt.clf()
            if plot:
                gs = fig.add_gridspec(2, 2, width_ratios = [1, 20], wspace = 0.05)
                ax_plot = fig.add_subplot(gs[1, :])
                ax_plot.plot(epochs, n_nodes, color = "k", marker = ".")
                ax_plot.set_xlabel("Killing Epochs [-]")
                ax_plot.set_ylabel("Number of Nodes [-]")
                ax_plot.axvline(epochs[1], color = "k", ls = "-.")
                ax_plot.text(epochs[1], n_nodes[0] / 2, f"first kill at epoch {epochs[1]}", 
                              rotation = "vertical", horizontalalignment = "right", 
                              verticalalignment = "center", color = "k")
                ax_plot.axvline(epochs[-1], color = "k", ls = "-.")
                ax_plot.text(epochs[-1], n_nodes[0] / 2, f"last kill at epoch {epochs[-1]}", 
                              rotation = "vertical", horizontalalignment = "right", 
                              verticalalignment = "center", color = "k")
                ax_plot.set_ylim(0, n_nodes[0])
                ax_plot.set_xlim(0, epochs[-1] + callback._log["delta_kill"][-1])
                ax_plot.grid(True)
            else:
                gs = fig.add_gridspec(1, 2, width_ratios = [1, 20], wspace = 0.05)
            
            ax_img = fig.add_subplot(gs[0, 1])
            ax_cb = fig.add_subplot(gs[0, 0])
                
            if not x_label:
                ax_img.set_xlabel("Features [-]")
            else:
                ax_img.set_xlabel(f"{x_label}")
                
            try:
                features = self.model_container.features
                features = np.char.replace(features, "_", "\_")
                ax_img.set_xticks(np.arange(len(features)))
                ax_img.set_xticklabels(features)
                x_lim = None
            except:
                pass
            
            img = np.empty([n_nodes[0] + 1, n_nodes[0]])
            for i in range(n_nodes[0] + 1):
                img[i] = np.array(np.sum(weights, axis = 0), ndmin = 2)
            
            if x_lim:
                ax_img.imshow(img, cmap = cmap, aspect = "auto", extent = (x_lim[0], x_lim[1], 0, 1))
            else:
                ax_img.imshow(img, cmap = cmap, aspect = "auto")
            ax_img.set_yticks([])
            
            # colorbar
            ticks = np.linspace(0, 1, len(n_nodes), endpoint=True)
            cb = fig.colorbar(cm.ScalarMappable(cmap = map), ticks = ticks, cax = ax_cb, aspect = 20)
            cb.ax.set_yticklabels(n_nodes)
            cb.ax.yaxis.set_ticks_position("left")
            cb.ax.yaxis.set_label_position("left")
            cb.ax.invert_yaxis()
            cb.set_label("Mask [I/O]") 
            
            fig.canvas.set_window_title("Feature Selection")
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
            for pos in range(len(self.model_container.model.layers)):
                if self.model_container.model.layers[pos].name == names[i]:
                    break
            layer_type = self.model_container.model.layers[pos].__class__.__name__
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
        test_model = self.model_container(inputs = self.model_container.model.inputs, outputs = self.model_container.model.get_layer(layer_name).output)
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
            test_model = self.model_container(inputs = self.model_container.model.inputs, outputs = self.model_container.get_layer(generator_layers[i]).output)
            feature_maps = self.model_container.test_model(x, model = test_model)
            ax = plt.subplot(len(generator_layers), 1, i + 1)
            plt.plot(feature_maps[0])
            plt.grid()
        fig.tight_layout()
        fig.show()
            
    
    # analyzers   
    def check_prediction(self, x, y, show_plot = False):
        x = self.model_container._convert_data(x, y)
        prediction = self.model_container.model.predict(x) * 100
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
                y_pred = self.model_container.test_model(self.x_train, self.model_container.y_train, model = self.model_container.model)
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
        feature_map = self.model_container.model.predict(array)
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
        if self.model_container.model_type == "spectral_net":
            random_pos = np.random.randint(0, len(self.x_train))
            
            label = self.model_container.labels[self.model_container.y_train[random_pos]]
            
            if self.model_container.classifier.architecture_type == "ann":
                signal = np.array(self.x_train[random_pos], ndmin = 2)
            else:
                signal = np.array(self.x_train[random_pos], ndmin = 3)
            
            mask = np.array(self.generator.x_train[random_pos], ndmin = 2)
            
            try:
                pred_model = self.model_container(inputs = self.model_container.model.inputs, outputs = self.model_container.model.get_layer('MaskedSignal').output)
                FM_signal = pred_model.predict(x = [signal, mask])
            except:
                raise NameError(f"Layer 'MaskedSignal' is not a part of self.model_container '{self.model_container.model_type}'")
            
            try:
                pred_model = self.model_container(inputs = self.model_container.model.inputs, outputs = self.model_container.model.get_layer("Generator").output)
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
            feature_map = self.model_container.model.predict(mask)
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