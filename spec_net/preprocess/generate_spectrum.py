import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . image_plot_conversion import extract_SDBS_data

class Spectrum:
    def __init__(self, label, group):
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
        self.x = np.round(np.linspace(4000, 400, 3600), 0)
        self.label = label
        self.group = group
        self.data = np.zeros(len(self.x))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.label}, {self.group})"

# setters
    def set_x(self, xmin, xmax, steps):
        self.x = np.round(np.linspace(xmin, xmax, steps), 0)
        
    def set_spectrum(self, label, group):
        self.label = label
        self.group = group
        
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

    def ifft(self, plot = None):
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
        if plot is None:
            plot = self.data
        ifft = np.fft.ifft(plot)
        return ifft
        
    def plot(self, *args, **kwargs):
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
        if "label" in kwargs:
            label = kwargs["label"]
        else:
            label = self.label
        
        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = "k"
        if len(args) != 0:
            plt.plot(self.x, args[0], label = f"{label}", color = color, linewidth = 1)
        else:
            plt.plot(self.x, self.data, label = f"{label}", color = color, linewidth = 1)
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(0, 100)
        plt.xlabel("Wavenumber [cm$^{-1}$]")
        plt.ylabel("Transmittance [$\%$]")
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