import pandas as pd
import numpy as np
from spec_net.preprocess.spectrum import Spectrum
from spec_net.preprocess.datahandling import CSVFile
import matplotlib.pyplot as plt

csv = CSVFile("realistic")

class Proteins:
    def __init__(self, protein_config):
        self.group = "proteins"
        protein_sum = np.sum(protein_config)
        self.albumin = protein_config[0] / protein_sum
        self.hemoglobin = protein_config[1] / protein_sum
        
    def get_config(self):
        return {"albumin": self.albumin, "hemoglobin": self.hemoglobin}
    
    def create(self, csv):
        ifft_merged_spectrum = None
        for label in self.get_config().keys():
            spectrum = Spectrum(label, self.group)
            spectrum.get_data(csv)
            ifft_spectrum = spectrum.ifft() * self.get_config()[label]
            if ifft_merged_spectrum is None:
                ifft_merged_spectrum = ifft_spectrum
            else:
                ifft_merged_spectrum += ifft_spectrum
        merged_spectrum = np.abs(np.fft.fft(ifft_merged_spectrum))
        self.data = merged_spectrum
        return self.data

    def plot(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)
        
class Carbohydrates:
    def __init__(self, carbohydrates_config):
        self.group = "carbohydrates"
        carbohydrates_sum = np.sum(carbohydrates_config)
        self.d_glucose = carbohydrates_config[0] / carbohydrates_sum
        self.d_mannose = carbohydrates_config[1] / carbohydrates_sum
        self.glycogen = carbohydrates_config[2] / carbohydrates_sum
        
    def get_config(self):
        return {"d_glucose": self.d_glucose, "d_mannose": self.d_mannose, "glycogen": self.glycogen}
    
    def create(self, csv):
        ifft_merged_spectrum = None
        for label in self.get_config().keys():
            spectrum = Spectrum(label, self.group)
            spectrum.get_data(csv)
            ifft_spectrum = spectrum.ifft() * self.get_config()[label]
            if ifft_merged_spectrum is None:
                ifft_merged_spectrum = ifft_spectrum
            else:
                ifft_merged_spectrum += ifft_spectrum
        merged_spectrum = np.abs(np.fft.fft(ifft_merged_spectrum))
        self.data = merged_spectrum
        return self.data

    def plot(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)

class Lipids:
    def __init__(self, lipids_config):
        self.group = "lipids"
        lipids_sum = np.sum(lipids_config)
        self.cholesterol = lipids_config[0] / lipids_sum
        self.lecithin = lipids_config[1] / lipids_sum
        self.testosterone = lipids_config[2] / lipids_sum
        
    def get_config(self):
        return {"cholestorol": self.cholesterol, "lecithin": self.lecithin, "testosterone": self.testosterone}
    
    def create(self, csv):
        ifft_merged_spectrum = None
        for label in self.get_config().keys():
            spectrum = Spectrum(label, self.group)
            spectrum.get_data(csv)
            ifft_spectrum = spectrum.ifft() * self.get_config()[label]
            if ifft_merged_spectrum is None:
                ifft_merged_spectrum = ifft_spectrum
            else:
                ifft_merged_spectrum += ifft_spectrum
        merged_spectrum = np.abs(np.fft.fft(ifft_merged_spectrum))
        self.data = merged_spectrum
        return self.data

    def plot(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)

class DNA:
    def __init__(self, DNA_config):
        self.group = "dna"
        dna_sum = np.sum(DNA_config)
        self.adenine = (DNA_config[0] + DNA_config[1]) / (4 * dna_sum)
        self.thymine = self.adenine
        self.cytosine = (DNA_config[2] + DNA_config[3]) / (4 * dna_sum)
        self.guanine = self.cytosine
        self.desoxyribose = 0.5
    
    def get_config(self):
        return {"adenine": self.adenine, "thymine": self.thymine, "cytosine": self.cytosine, "guanine": self.guanine, "2_desoxy_d_ribose": self.desoxyribose}

    def create(self, csv):
        ifft_merged_spectrum = None
        for label in self.get_config().keys():
            spectrum = Spectrum(label, self.group)
            spectrum.get_data(csv)
            ifft_spectrum = spectrum.ifft() * self.get_config()[label]
            if ifft_merged_spectrum is None:
                ifft_merged_spectrum = ifft_spectrum
            else:
                ifft_merged_spectrum += ifft_spectrum
        merged_spectrum = np.abs(np.fft.fft(ifft_merged_spectrum))
        self.data = merged_spectrum
        return self.data

    def plot(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)
        
class RNA:
    def __init__(self, RNA_config):
        self.group = "rna"
        rna_sum = np.sum(RNA_config)
        self.adenine = (RNA_config[0] + RNA_config[1]) / (4 * rna_sum)
        self.uracil = self.adenine
        self.cytosine = (RNA_config[2] + RNA_config[3]) / (4 * rna_sum)
        self.guanine = self.cytosine
        self.ribose = 0.5
    
    def get_config(self):
        return {"adenine": self.adenine, "uracil": self.uracil, "cytosine": self.cytosine, "guanine": self.guanine, "d_ribose": self.ribose}
    
    def create(self, csv):
        ifft_merged_spectrum = None
        for label in self.get_config().keys():
            spectrum = Spectrum(label, self.group)
            spectrum.get_data(csv)
            ifft_spectrum = spectrum.ifft() * self.get_config()[label]
            if ifft_merged_spectrum is None:
                ifft_merged_spectrum = ifft_spectrum
            else:
                ifft_merged_spectrum += ifft_spectrum
        merged_spectrum = np.abs(np.fft.fft(ifft_merged_spectrum))
        self.data = merged_spectrum
        return self.data

    def plot(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)
        
class NucleicAcid:
    def __init__(self, config, proliferation_factor):
        self.group = "nucleic_acid"
        self.proliferation_factor = [0.5 + 0.1 * proliferation_factor, 0.5 - 0.1 * proliferation_factor]
        self.dna = DNA(config)
        self.rna = RNA(config)
    
    def __repr__(self):
        return "NucleicAcid"
    
    def get_config(self):
        dna_config = scale_dict(self.dna.get_config(), self.proliferation_factor[0])
        rna_config = scale_dict(self.rna.get_config(), self.proliferation_factor[1])
        for key in rna_config:
            if key in dna_config.keys():
                dna_config[f"{key}"] += rna_config[f"{key}"]
            else:
                dna_config[f"{key}"] = rna_config[f"{key}"]
        config = dna_config
        return config
    
    def create(self, csv):
        dna = self.dna.create(csv)
        rna = self.rna.create(csv)
        ifft_dna = np.fft.ifft(dna)
        ifft_rna = np.fft.ifft(rna)
        spectrum = np.fft.fft(ifft_dna * self.proliferation_factor[0] + ifft_rna * self.proliferation_factor[1])
        self.data = spectrum
        return self.data
    
    def plot(self):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)
    
class Tissue:
    def __init__(self, nucleic_acid, proteins, carbohydrates, lipids, weights = [0.25, 0.6, 0.11, 0.04]):
        self.group = "tissue"
        self.nucleic_acid = nucleic_acid
        self.proteins = proteins
        self.carbohydrates = carbohydrates
        self.lipids = lipids
        self.weights = weights / np.sum(weights)
    
    def get_config(self):
        config = {}
        i = 0
        for obj in (self.nucleic_acid, self.proteins, self.carbohydrates, self.lipids):
            dict = obj.get_config()
            dict = scale_dict(dict, self.weights[i])
            i += 1
            config.update(dict)
        return config
    
    def create(self, csv):
        nucleic_acid = self.nucleic_acid.create(csv)
        proteins = self.proteins.create(csv)
        carbohydrates = self.carbohydrates.create(csv)
        lipids = self.lipids.create(csv)
        ifft_nucleic_acid = np.fft.ifft(nucleic_acid)
        ifft_proteins = np.fft.ifft(proteins)
        ifft_carbohydrates = np.fft.ifft(carbohydrates)
        ifft_lipids = np.fft.ifft(lipids)
        spectrum = np.fft.fft(ifft_nucleic_acid * self.weights[0] + ifft_proteins * self.weights[1] + ifft_carbohydrates * self.weights[2] + ifft_lipids * self.weights[3])
        self.data = spectrum
        return self.data
    
    def plot(self, **kwargs):
        x = np.round(np.linspace(4000, 400, 3600), 0)
        plt.plot(x, self.data, **kwargs)
        plt.xlim(4000, 400)
        plt.ylim(0, 100)
        plt.ylabel("Transmittance T [$\%$]")
        plt.xlabel("Wavenumber $\\tilde{\\nu}$ [cm$^{-1}$]")

def scale_dict(dict, scalar):
    for entry in dict.keys():
        dict[f"{entry}"] = dict[f"{entry}"] * scalar
    return dict

def generate_dataset(size):
    data = []
    label = []
    for i in range(size):
        human_config = np.random.uniform(0, 1, size=([4]))
        weights = [np.random.uniform(0.55, 0.65), np.random.uniform(0.2, 0.3), np.random.uniform(0.05, 0.2), np.random.uniform(0.05, 0.2)]
        proteins = Proteins(human_config[0:2])
        carbohydrates = Carbohydrates(human_config[0:3])
        lipids = Lipids(human_config[0:3])
        for proliferation in (0, 0.2, 0.4, 0.6, 0.8, 1):
            weights[1] = weights[1] * (1 + proliferation)
            nucleic_acid = NucleicAcid(human_config, proliferation)
            tissue = Tissue(nucleic_acid, proteins, carbohydrates, lipids, weights = weights)
            tissue.create(csv)
            if proliferation <= 0.5:
                label.append("healthy")
            else:
                label.append("diseased")
            data.append(tissue.data)
    data = np.array(data).real
    label = np.array(label)
    return data, label

def plot_signal(signal, **kwargs):
    x = np.round(np.linspace(4000, 400, 3600), 0)
    plt.plot(x, signal, **kwargs)
    plt.xlim(4000, 400)
    plt.ylim(0, 100)
    plt.ylabel("Transmittance T [$\%$]")
    plt.xlabel("Wavenumber $\\tilde{\\nu}$ [cm$^{-1}$]")
