import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .image_plot_conversion import extract_SDBS_data
from .data import resample
from ..utils.time import timer
from spec_net.preprocess.filter import Filter
from scipy.interpolate import interp1d

class Spectrum:
    def __init__(self, data = None, label = None, x = None, group = None, metric = None):
        """
        Class to inspect, modify and transform spectral plot data.

        Parameters
        ----------
        label : str
            The molecule name of the spectral plot data.
        group : str
            The functional group of the spectral plot data.
        source : str
            States the path of the source csv-file.

        Returns
        -------
        None.

        """
        if x is None:
            self.x = np.round(np.linspace(4000, 400, len(self.data)), 0)
        else:
            if isinstance(x, list):
                self.x = np.round(np.linspace(x[0],x[1], len(self.data)), 0)
            else:
                self.x = x
        self.label = label
        self.group = group
        self.data = data
        self.metric = metric
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.label}, {self.group})"

# setters
    def set_x(self, xmin, xmax, steps):
        self.x = np.round(np.linspace(xmin, xmax, steps), 0)
        self.data = resample(self.data, len(self.x))
        
    def set_label(self, label, group):
        self.label = label
        self.group = group
    
    def set_metric(self, metric):
        self.metric = metric
        
# getters    
    def generate_data(self, path):
        """
        Generates the spectrum from given pictures [.png] with the given spectra.

        Parameters
        ----------
        path : str
            Path that leads to the following folder structure: [self.group]-->[self.label].png.
        """
        self.data = extract_SDBS_data(f"{path}/{self.group}/{self.label}.png")
    
    def get_data(self, csv):
        """
        Scans through the specified csv file and searches for the suited plot.
        
        Parameters
        ----------
        csv_data : pd-df
            CSV data in which the plots are stored.

        Returns
        -------
        None.

        """
        df = csv.data
        self.data = np.array(df.loc[(df["label"] == self.label) & (df["group"] == self.group)])[0, 2:].astype(float)
    
    def save_data(self, csv):
        df = csv.data
        if not df.loc[(df["label"] == self.label) & (df["group"] == self.group)].empty:
            print("Already part of CSV File")
        else:       
            if np.sum(self.data) == 0:
                raise Exception("Data not generated yet.")
            else:
                cols = ["label", "group"] + [str(self.x[i]) for i in range(len(self.x))]
                data = np.append([self.label, self.group], self.data)
                data = pd.DataFrame(data.reshape([1, len(data)]), columns = cols)
                df = pd.concat([df, data], ignore_index = True)
                df.to_csv(csv.csv_path, index = False)
                print(f"Saved {self.label} from {self.group} at {csv.csv_path}.")
    
    def filter(self, type = "savitzky_golay", *args, **kwargs):
        self.data = Filter().filter(self.data, type, *args, **kwargs)
    
    @timer
    def ifft(self):
        """
        Inverse transforms the spectral plot data.

        Parameters
        ----------
        plot : np-array (1D: float), optional
            Only needed if molecule differs from initiallized molecule. The default is None.

        Returns
        -------
        ifft : np-array (1D: float)
            The inverse FT of the plot (original measured signal before using FT in FTIR-spectroscopy).

        """
        self.data = np.fft.ifft(self.data)
    
    def transmittance(self):
        if self.metric == "transmittance":
            print("Spectrum is already in transmission mode.")
        else:
            self.set_metric("transmittance")
            self.data = 10**(2 - self.data)

    def absorbance(self):
        if self.metric == "absorbance":
            print("Spectrum is already in absorption mode.")
        else:
            self.set_metric("absorbance")
            self.data = 2 - np.log10(self.data)
    
    @staticmethod
    def get_derivation(data, n = 1):
        for i in range(n):
            l_data_offset = np.append(data[0], data[:-1])
            r_data_offset = np.append(data[1:], data[-1])
            data = 1/2 * (r_data_offset - l_data_offset)
        return data
    
    def get_extrema(self):
        extrema = {"minima": [], "maxima": []}
        derivative = self.get_derivation(self.data)
        for i in range(len(derivative)):
            if i == 0 or i == len(derivative) - 1:
                continue
            if derivative[i - 1] > 0 and derivative[i] < 0:
                extrema["maxima"].append(i)
            elif derivative[i -1] < 0 and derivative[i] > 0:
                extrema["minima"].append(i)
            else:
                pass
        return extrema
    
    def get_global_minima(self, window):
        delta = int((window - 1) / 2)
        extrema = self.get_extrema()
        minima = extrema["minima"]
        n = len(minima)
        length = len(self.data) - 1
        glob, loc = [], []
        
        for i in range(0, n):
            loc.append(minima[i])
            if i  == n - 1:
                glob.append(loc)
            else:
                if np.abs(minima[i] - minima[i + 1]) >= delta:
                    glob.append(loc)
                    loc = []
        
        loc_min = [0]
        for loc in glob:
            pos_min = loc[np.argmin(self.data[loc])]
            loc_min.append(pos_min)
        loc_min.append(length)
        return loc_min
    
    def get_baseline(self, window = 401, method = "cubic", plot = False, optimize = False, e = 0.005):
        
        loc_min = np.array(self.get_global_minima(window = window))
        extrema = np.array(self.get_extrema()["minima"])
        baseline = interp1d(self.x[loc_min], self.data[loc_min], kind = method)(self.x)
        if optimize:
            while True:
                min_val = float("inf")
                extrema = extrema[np.isin(extrema, loc_min, invert = True)]
                iterated = False
                for loc in extrema:
                    if baseline[loc] > self.data[loc]:
                        if np.abs(baseline[loc] - self.data[loc]) < min_val:
                            min_val, min_loc = self.data[loc], loc
                            iterated = True
                if iterated == False:
                    break
                loc_min = np.append(loc_min, min_loc)
                baseline = interp1d(self.x[loc_min], self.data[loc_min], kind = method)(self.x)
            while True:
                pos_area = np.where(self.data >= baseline)
                neg_area = np.where(self.data < baseline)
                pos = self.data[pos_area] - baseline[pos_area]
                neg = baseline[neg_area] - self.data[neg_area]
                error = np.sum(neg) / (np.sum(neg) + np.sum(pos))
                if e < error:
                    loc = np.argwhere(baseline - self.data == np.amax(neg))
                    loc_min = np.append(loc_min, loc)
                    baseline = interp1d(self.x[loc_min], self.data[loc_min], kind = method)(self.x)
                else:
                    break
        if plot:
            self.plot(baseline, label = f"baseline")
            plt.scatter(self.x[loc_min], self.data[loc_min])
            plt.legend()
        return baseline
    
    def baseline_correction(self, window = 401, method = "cubic", optimize = True):
        baseline = self.get_baseline(window = window, method = method, optimize = optimize)
        self.data = np.where((self.data - baseline) >= 0, self.data-baseline, 0)
    
    def derivative(self, n = 1):
        self.data = self.get_derivation(self.data, n = n)
        
    def polyfit(self, n = 1):
        self.data = np.poly1d(np.polyfit(self.x, self.data, deg = n))

    def baseline_als(self, lam, p, n_iter=10):
        # algorithm from "Asymmetric Least Squares Smoothing" - Eilers, Boelens 2005
        """
        Asymmetric Least Squares Smoothing.
        
        Parameters
        ----------
        lam : float
            Smoothness parameter (10^2 <= lam <= 10^9).
        p : float
            Asymmetry parameter (0.001 <= p <= 0.1)
        """
        L = len(self.data)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(n_iter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * self.data)
            w = p * (self.data > z) + (1 - p) * (self.data < z)
        self.data = z
        
    def multiplot(self, data, labels):
        # ensure absorbance data
        height = 0
        for i in range(len(data)):
            plt.plot(self.x, data[i] + height, color = "k")
            plt.text(self.x[-1] - 20, data[i, -1] + height, f"{labels[i]}", horizontalalignment = "left")
            height += np.amax(data[i])
        plt.yticks([])
        plt.ylabel("Absorbance [a.u.]")
        plt.xlabel("Wavenumber [cm$^{-1}$]")
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(0, height * 1.1)
        plt.tight_layout()
        
    def plot(self, data = None, metric = "transmittance", *args, **kwargs):
        """
        Plots data in the common spectroscopic way.

        Parameters
        ----------
        *args : np-array (1D: float)
            Plot data. Only needed if molecule differs from initiallized molecule.
        **kwargs : str
            Label data. Only needed if molecule differs from initiallized molecule.

        Returns
        -------
        None.

        """
        plt.figure(f"{self.label}")
        
        if "label" not in kwargs:
            kwargs["label"] = self.label
        
        if data is None:
            data = self.data
        
        if self.metric is not None:
            metric = self.metric
        
        plt.plot(self.x, data, linewidth = 1, **kwargs)
        try:
            plt.plot(self.x, self.baseline, color = "r")
        except:
            pass
        plt.xlim(self.x[0], self.x[-1])
        
        plt.xlabel("Wavenumber [cm$^{-1}$]")
        if metric == "transmittance":
            ax = plt.gca()
            ax.yaxis.set_major_formatter(PercentFormatter())
            plt.ylim(0, np.amax(data) * 1.05)
            plt.ylabel("Transmittance")
        elif metric == "absorbance":
            plt.ylabel("Absorbance")
            plt.ylim(0, np.amax(data) * 1.05)
        elif metric == "raman":
            # plt.ylim(np.amin(self.data) * 1.05, np.amax(self.data) * 1.05)
            plt.ylabel("Rel. Scattering Intensity")
        plt.legend()
        plt.grid(True)
    
    def plot_DFT(self, signal = None, n = None, d_type = "abs"):
        """
        Plots the DFT of the original data.

        Parameters
        ----------
        signal : np-array, optional
            Plot data. Only needed if molecule differs from initiallized molecule. The default is None.
        n : int, optional
            Sampling points. The default is None.
        d_type : str, optional
            States the information that is to be plotted. The default is "abs".

        Returns
        -------
        None.

        """
        
        if signal is None:
            signal = self.plot
        
        if n is None:
            n = len(signal)
        
        X = np.fft.fft(signal, n = n)
        f = np.arange(-len(signal) / 2, len(signal) / 2, len(signal) / n)
        
        if len(X) % 2 == 0:
            X = np.append(X[int(n / 2) : n], X[0 : int(n / 2)])
        else:
            X = np.append(X[int(n / 2) : n], X[0 : int(n / 2)])
        
        if d_type == "abs":
            X = np.abs(X)
            plt.title("Magnitude spectrum")
            plt.ylabel("|X(f)|")
        elif d_type == "phase":
            plt.title("Phase spectrum")
            threshold = np.amax(np.abs(X)) / 10000
            X[X <= threshold] = 0
            X = np.angle(X)
            plt.ylabel("$\\angle$ X(f) [rad]")
        elif d_type == "real":
            X = X.real
            plt.ylabel("real part")
        elif d_type == "imag":
            X = X.imag
            plt.xlabel("imag part")
        else:
            pass
        
        plt.plot(f, X * 2 / len(signal), label = f"DFT of {self.label}: {n}-sampled")
        plt.legend(loc = "upper right")
        plt.xlabel("f [Hz]")
        plt.xlim(-len(signal) / 2, len(signal) / 2)
        plt.grid(True)

class ManipulateSpectrum(Spectrum):
    def __init__(self, *args, **kwargs):
        """
        Class to inspect, modify and transform spectral plot data.

        Parameters
        ----------
        label : str
            The molecule name of the spectral plot data.
        group : str
            The functional group of the spectral plot data.
        csv_path : str
            States the path of the csv file.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        
    def mask_plot(self):
        """
        Masks original plot.

        Returns
        -------
        masked_plot : np-array (1D: float)
            Masked Plot.

        """
        masked_plot = self.plot * self.data_handling.bool_mask(self.plot)
        return masked_plot
    
    def add_noise(self, SNR = 20000):
        """
        Adds white noise.

        Parameters
        ----------
        SNR : float, optional
            Signal to noise ratio. The default is 20000.

        Raises
        ------
        NameError
            If plot can not be found.

        Returns
        -------
        noisy_plot : np-array (1D: float)
            Noisy plot signal.

        """
        try:    
            self.noise = np.random.normal(0, 1/np.sqrt(2 * SNR), len(self.plot))
            noisy_plot = self.plot + self.noise
        except:
            raise NameError("Can not find plot to add noise to.")
        return noisy_plot
        
    def merge_plots(self, labels, iterations = 1000, show_plot = False):
        """
        Generates a simplified spectrogram. Takes multiple plots and merges them together.

        Parameters
        ----------
        labels : list (str)
            States the molecules or plots that shall be merged together.
        iterations : int, optional
            States how many steps it takes to fully convert the n-th molecule into the n+1-th molecule. The default is 1000.
        show_plot : bool, optional
            Shows image if 'True'. The default is False.

        Returns
        -------
        transitions : np-array (2D: float)
            Image of transitional spectrogram.

        """
        plots = []
        for i in range(0, len(labels)):
            label, group, plot = self.get_plot(label = labels[i])
            plots.append(plot)
    
        transitions = np.empty([len(plot), iterations * (len(labels) - 1)])
        
        for plot_index in range(0, len(plots) - 1):
        
            plot_1 = np.array(plots[plot_index])
            plot_2 = np.array(plots[plot_index + 1])
        
            transition = np.empty([len(plot_1), iterations])
            for i in range(0, iterations):
                plot = i / iterations * (plot_2 - plot_1) + plot_1
                transition[ : , i] = plot
            transitions[ : , plot_index * iterations : (plot_index + 1) * iterations] = transition   
        
        if show_plot == True:
            plt.figure("Transitional Spectogram")
            plt.imshow(transitions)
            y = np.linspace(self.x[0], self.x[-1], 10)
            y_pos = np.linspace(0, transitions.shape[0], 10)
            plt.yticks(y_pos, y)
            plt.ylabel("Wavenumber [cm$^-1$]")
            plt.xlabel("Translation")
            plt.title("Transitional Spectrogram [-]")
        
        return transitions
    
    def water_disturbance(self, concentration, showPlot = False):
        """
        Linear function of disturbance by water influences. In reality, this is a non-linear process in dilute solutions due to e.g. changing of hydrogen bonds.
        
        Parameters
        ----------
        concentration : float
            Determines the relative amount of the observed substance.
        show_plot : bool, optional
            Shows image if 'True'. The default is False.
        
        Returns
        -------
        disturbed : np-array (1D: float)
            Data of mixture between water and the molecule.
            
        """
        ifft_plot = self.ifft()
        label, group, plot = self.get_plot(label = "water")
        ifft_water = self.ifft(plot)
        ifft_disturbed = np.add(concentration * ifft_plot, (1 - concentration) * ifft_water)
        disturbed = np.abs(np.fft.fft(ifft_disturbed))
        
        if showPlot == True:
            plt.figure("Water Disturbance")
            plt.title("Water Disturbance")
            self.plot_data(disturbed, label = f"{concentration * 100} % {self.label}")
            
        return disturbed
    
    def randomizeData(self, group):
        return