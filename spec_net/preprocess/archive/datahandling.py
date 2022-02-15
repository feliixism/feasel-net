import os
from .spectrum import Spectrum, ManipulateSpectrum
# from .tissue import generate_dataset

import numpy as np
import pandas as pd

class CSVFile:
    def __init__(self, csv_file):
        """
        Class to load, save spectral data given by the csv-path.

        Parameters
        ----------
        csv_path : str
            Sets path to access csv-data of spectra.

        Returns
        -------
        CSVFile class object.

        """
        self.csv_file = csv_file
        self.path = "data/csv/"
        self.csv_path = self.path + self.csv_file + ".csv"
        try:
            self.data = self.get_data()
        except:
            self.new_file()
            
    def __repr__(self):
        return f"{self.__class__.__name__}(Data stored in '{self.csv_path}')"
    
    def new_file(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        cols = ["label", "group"] + [str(x[i]) for i in range(len(x))]
        df = pd.DataFrame([], columns = cols)
        df.to_csv(self.csv_path, index = False)
        self.data = df
    
    def get_data(self):
        """
        Automatically initiated when calling class 'CSVFile'.
        
        Raises
        ------
        NameError
            Could not find data at specified path.

        Returns
        -------
        data : DataFrame
            Database with infrared spectra of various molecules.

        """
        try:
            csv_data = pd.read_csv(self.csv_path)
            print(f"Reading data stored at: {self.csv_path}")
            return csv_data
        except:
            raise NameError(f"Could not find data at '{self.csv_path}'")
            
    
    
    def generate_CSV_from_all_files(self, spectra_folder):
        """
        Searches for image-data from the SDBS-database (PNG) in the given spectra-folder and uses this data to generate a list with plot information of all stored images.

        Parameters
        ----------
        spectra_folder : str
            States the path of the folder that is being searched for image data.

        Returns
        -------
        None.

        """
        self.spectra_folder = spectra_folder
        for group in os.listdir(f"{spectra_folder}"):
            for file in os.listdir(f"{spectra_folder}/" + group):
                self.data = self.get_data()
                df = self.data
                label = file.split(".")[0]
                if not df.loc[(df["label"] == label) & (df["group"] == group)].empty:
                    print("Already part of CSV File")
                else:
                    spectrum = Spectrum(label, group)
                    spectrum.generate_data(spectra_folder)
                    spectrum.save_data(self)
        
    def generate_random_dataset(self, concentrations = 100):
        """
        Generates a random dataset on basis of the given csv-file. 
        White noise (SNR = 0.1) and a superpositioned water disturbance is used to augment the data.
        The randomness is implemented by shuffling the data in the end.

        Parameters
        ----------
        concentrations : int, optional
            States how many different mixtures of water are being generated per molecule. The default is 100.

        Returns
        -------
        plots : np-array (2D: str)
            Plot values of various spectra. First dimension is the index of the plot and the second is the wavenumber.
        labels : np-array (2D: str)
            Labels of various spectra corresponding to their values stored in 'X'. First dimension is the index of the plot and the second is the label.

        """
        from .spectrum import Spectrum, ManipulateSpectrum
        labels = []
        plots = []
        df = np.array(pd.read_csv(self.csv_path))
        steps = concentrations
        for i in range(0, len(df)):
            if df[i, 0] == "water":
                    continue
            for j in range(0, steps):    
                labels.append(df[i, 0])
                spectrum = ManipulateSpectrum(df[i, 0], df[i, 1], self.csv_path)
                spectrum.data = self.data
                spectrum.get_plot(label = f"{df[i, 0]}")
                spectrum.add_noise(0.1)
                disturbed_plot = spectrum.water_disturbance(j / steps)
                plots.append(disturbed_plot)
        labels = np.array(labels)
        plots = np.array(plots)
        plots, labels = self.shuffle_data(plots, labels)
        return plots, labels
    
    def shuffle_data(self, X, y):
        """
        Shuffeling the data.

        Returns
        -------
        X : np-array (2D: str)
            Plot values of various spectra. First dimension is the index of the plot and the second is the wavenumber.
        y : np-array (2D: str)
            Labels of various spectra corresponding to their values stored in 'X'. First dimension is the index of the plot and the second is the label.

        """
        y = y.reshape([len(y), 1])
        concatenated = np.concatenate((y, X), axis = 1)
        np.random.shuffle(concatenated)
        y = concatenated[ : , 0 : 1]
        X = concatenated[ : , 1 : ].astype(float)
        return X, y

class NumpyFile:
    def __init__(self, numpy_file):
        """
        Class to load, save spectral data given by the numpy-file.

        Parameters
        ----------
        numpy_file : str
            Sets path to access numpy-data of spectra.

        Returns
        -------
        NumpyFile class object.

        """
        self.numpy_file = numpy_file
        self.path = "data/npy/"
        self.numpy_path = self.path + self.numpy_file + "/"
        
    def __repr__(self):
        return f"{self.__class__.__name__}(Data stored in '{self.numpy_path}')"
    
    def get_data(self):
        """
        Initiallized when calling 'NumpyFile' object. Resets all prior manipulation of the data as well.

        Returns
        -------
        X : np-array (2D: str)
            Plot values of various spectra. First dimension is the index of the plot and the second is the wavenumber.
        y : np-array (2D: str)
            Labels of various spectra corresponding to their values stored in 'X'. First dimension is the index of the plot and the second is the label.

        """
        y = np.load(self.numpy_path + "base_y.npy")
        X = np.load(self.numpy_path + "base_X.npy")
        return X, y
    
    def save_data(self, X, y):
        """
        Storing a new numpy-file at the given path.

        Parameters
        -------
        X : np-array (2D: str)
            Plot values of various spectra. First dimension is the index of the plot and the second is the wavenumber.
        y : np-array (2D: str)
            Labels of various spectra corresponding to their values stored in 'X'. First dimension is the index of the plot and the second is the label.
        filename : str
            States the path on which the data is to be stored.
        """
        cwd = os.getcwd()
        if os.path.isdir(cwd + "/" + self.numpy_path) == False:
            os.mkdir(cwd + "/" + self.numpy_path)
        X = X.astype(float)
        y = y.astype(str)
        np.save(self.numpy_path + "base_y.npy", y)
        np.save(self.numpy_path + "base_X.npy", X)
        print("Saved new numpy base file at {self.path + self.numpy_file}.")
               
    def fingerprint_region(self):
        """
        The region of interest (ROI) of the spectrum is set to the so called fingerprint region (wavenumber: 1500 - 500 1/cm).
        
        Returns
        -------
        X : np-array (2D: str)
            Plot values of various spectra. First dimension is the index of the plot and the second is the wavenumber.
        y : np-array (2D: str)
            Labels of various spectra corresponding to their values stored in 'X'. First dimension is the index of the plot and the second is the label.

        """
        self.X = self.X[ : , 2500:3500]
        return self.X, self.y
    

     
    def bool_mask(self, n_pos = 4, max_width = 20):
        """
        Randomly generates a boolean mask with the length of the spectral plots.

        Parameters
        ----------
        n_pos : int, optional
            Determines the number of potential lasers. The default is 4.
        max_width : int, optional
            Determines the maximal bandwidth of each laser. The default is 20.

        Returns
        -------
        mask : bool
            Randomly generated boolean mask with the length of the spectral plots.

        """
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(1, len(self.X))
        self.mask = np.zeros([self.X.shape[1]], dtype = bool)
        positions = 0
        while positions < n_pos:
            position = np.random.randint(0, self.X.shape[1])
            width = np.random.randint(2, max_width)
            try:
                self.mask[int(position - width / 2) : int(position + width / 2)] = True
                positions += 1
            except:
                positions = positions
        return self.mask
    
    def sparse_spectrum(self, mask = None):
        """
        Randomly generates a boolean mask with the length of the spectral plots and cuts the .

        Parameters
        ----------
        mask : boolean array, optional
            Mask to limit the original plot. The default is None.

        Returns
        -------
        sparse_plots : np-array (2D: str)
            Plot values of various spectra. First dimension is the index of the plot and the second is the wavenumber.
        y : np-array (2D: str)
            Labels of various spectra corresponding to their values stored in 'X'. First dimension is the index of the plot and the second is the label.

        """
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(1, len(self.X))
        if mask is None:
            self.mask = self.bool_mask()
        sparse_plots = []
        for i in range(0, self.X.shape[0]):
            sparse_plots.append(self.X[i][mask,...])
        sparse_plots = np.array(sparse_plots)
        return sparse_plots, self.y