import os
import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, filename, type = "csv"):
        self.filename = filename.split(".")[0]
        self.root_path = "/".join(os.getcwd().split("\\")[:-2])
        self.file_path = self.root_path + f"/data/{type}/" + filename
        
        if not os.path.isfile(self.file_path):
            raise NameError(f"Can not find the file at '{self.file_path}'.")
        
        self.type = type
        
    def get_pandas(self):
        if self.type == "excel":
            self.df = pd.read_excel(self.file_path)
            return self.df
        
    def filter(self, keys = None, values = None, item_axis = 0):
        if keys is None and values is None:
            raise NameError("Do not have enough filter information.")
        
        if not hasattr(self, "df"):
            raise ValueError("Do not have any DataFrame object.")
        
        else:
            if keys is None:
                raise NameError("Please enter a valid key.")
            elif values is None:
                return self.df[keys]
            else:
                if isinstance(keys, list):
                    df = self.df
                    try:
                        for i in range(len(keys)):
                            df = df[df[f"{keys[i]}"] == f"{values[i]}"]
                        return df
                    except:
                        raise NameError("Number of 'keys' doesn't match number of 'values'.")
                else:
                    return self.df[self.df[keys] == values]

    def write_numpy(self, dataframe = None, filename = None, label_idx = 0):
        path = self.root_path + "/data/npy/" + filename
        if not os.path.exists(path):
            os.makedirs(path)
        if dataframe is None:
            labels = self.df.iloc[:,:label_idx].to_numpy(str)
            data = self.df.iloc[:,label_idx:].to_numpy(float)
        else:
            labels = dataframe.iloc[:,:label_idx].to_numpy(str)
            data = dataframe.iloc[:,label_idx:].to_numpy(float)
        if filename is None:
            filename = self.filename
        np.save(path + "/base_X.npy", data)
        np.save(path + "/base_y.npy", labels)
        print(f"Saved npy-files at: '{path}'.")

        
# test = DataHandler("Raman/nuclei_collagen_all.xlsx", type = "excel")
# test.get_pandas()
# hello = test.filter(keys = ["Stage"], values = ["Proliferation"])
# test.write_numpy(hello, "proliferation", label_idx = 4)

# test.df[test.df["Finding"] == "Control"]
# test.df[test.df["Occurence"] == "Cytoplasm"]

# list1 = ("Finding", "Occurence")
# list2 = ("Diseased", "Nuclei")
# df = test.df
# for i in range(len(list1)):
#     print(i)
#     df = df[df[f"{list1[i]}"] == f"{list2[i]}"]
#     test.df[(test.df[f"{list1[i]}"] == f"{list2[i]}") for i in np.arange(len(list1))]]
# test.df[(test.df["Finding"] == "Diseased") & (test.df["Occurence"] == "Nuclei")]

# a = test.get_pandas()

# a.filter(items = "Findings", axis = 0)
# a = np.load("C:\\Users\\itofischer\\Desktop\\SpectralAnalysis\\data\\npy\\nuclei_filtered\\base_X.npy")

