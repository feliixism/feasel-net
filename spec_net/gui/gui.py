import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.filedialog as fd
import pandas as pd
from .additional_widgets.scrollable_image import ScrollableImage, PlotImage
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

from ..preprocess.spectrum import ManipulateSpectrum

from ..preprocess import image_plot_conversion as i2p
from ..preprocess.datahandling import CSVFile
import os

import numpy as np

class SpecApp(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.output = []
        self.create_frame()
        self.create_widgets()
        self.create_canvas()
        self.create_menu()
        self._switched = False
        self.csv_path = None
        self.x = np.round(np.linspace(4000, 400, 3600), 0)
                
    def create_menu(self):
        menu = tk.Menu(self.master, tearoff=False)
        self.master.config(menu = menu)
        
        fileMenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label = "File", menu = fileMenu)
        
        fileMenu.add_command(label = "Load File", command = self.load_csv)
        fileMenu.add_command(label = "New File", command = self.save_csv)
        fileMenu.add_separator()
        fileMenu.add_command(label = "Exit", command = self.master.destroy)
        
        editMenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label = "Edit", menu = editMenu)
        editMenu.add_command(label = "Undo")
        editMenu.add_command(label = "Remake")
        editMenu.add_separator()
        editMenu.add_command(label = "Load Image", command = self.get_file)
        editMenu.add_command(label = "Load Plot", command = self.load_plot)
        editMenu.add_command(label = "Save Plot", command = self.save_plot)
        editMenu.add_command(label = "FFT", command = self.fft)
        editMenu.add_command(label = "Water Disturbance", command = self.water_disturbance)
        editMenu.add_command(label = "Convert", command = self.convert_image)
        editMenu.add_command(label = "Mask")
        
        viewMenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label = "View", menu = viewMenu)
        viewMenu.add_command(label = "CSV", command = self.get_csv_groups)
        viewMenu.add_command(label = "Switch", command = self.switch_positions)

#file menu:
    
    def create_frame(self):
        self.window_frame = tk.Frame(self.master, bg = "white", relief = tk.GROOVE)
        self.window_frame.pack(side = "top", anchor = tk.NW, fill = "both", expand = True, padx = 5, pady = 5)
    
    def get_csv_groups(self):
        self.df = pd.read_csv(self.csv_path)
        
        self.groups = np.unique(self.df.values[:, 1])     
        self.groups = [i.replace("[", "").replace("]", "").replace("'", "") for i in self.groups]
        self.group_drop.configure(values = self.groups)
    
    def get_csv_labels(self, event):
        mask = self.df['group'] == self.group_drop.get()
        self.labels = self.df[mask]
        self.labels = np.unique(self.labels.values[:, 0])
        self.labels = [i.replace("[", "").replace("]", "").replace("'", "") for i in self.labels]
        self.label_drop.configure(values = self.labels)
        self.label_drop.set(self.labels[0])
    
    def create_widgets(self):
        self.labels = ["-"]
        self.groups = ["-"]
        self.widget_width = 250
        self.widgets_frame = tk.Frame(self.window_frame, relief = tk.GROOVE, borderwidth = 1, width = self.widget_width)
        self.widgets_frame.pack(side = "right", anchor = tk.NE, fill = "y")
        self.widgets_frame.pack_propagate(0)
        
        self.widget_label = tk.Label(self.widgets_frame, text = "Options:", font = "Arial 12 bold")
        self.widget_label.pack(side = "top", anchor = tk.NW, padx = 15)
        
        self.widget_buttons = tk.Frame(self.widgets_frame, relief = tk.GROOVE, borderwidth = 1)
        self.widget_buttons.pack(side = "top", anchor = tk.N)
        
        self.convert = tk.Button(self.widget_buttons, text = "Convert", width = 12, height = 1)
        self.convert["command"] = self.convert_image
        self.convert.grid(row = 0, column = 0, padx = 5, pady = 5)
        
        self.switch = tk.Button(self.widget_buttons, text = "Switch", width = 12, height = 1)
        self.switch["command"] = self.switch_positions
        self.switch.grid(row = 0, column = 1, padx = 5, pady = 5)
        
        self.save = tk.Button(self.widget_buttons, text = "Save Plot", width = 12, height = 1, command = self.save_plot)
        self.save.grid(row = 1, column = 0, padx = 5, pady = 5)

        self.load = tk.Button(self.widget_buttons, text="Load Plot", width = 12, height = 1)
        self.load["command"] = self.load_plot
        self.load.grid(row = 1, column = 1, padx = 5, pady = 5)
        
        divider = tk.Label(self.widgets_frame, text = "________________", font = "Arial 8")
        divider.pack(side = "top", pady = 5)

        self.choose_group_label = tk.Label(self.widgets_frame, text = "Group:", font = "Arial 10 bold")
        self.choose_group_label.pack(side = "top", anchor = tk.NW, padx = 15)
       
        self.group_drop = ttk.Combobox(self.widgets_frame, values = self.groups)
        self.group_drop.set(["-"])
        self.group_drop.bind("<<ComboboxSelected>>", self.get_csv_labels)
        self.group_drop.pack(side = "top", anchor = tk.N, padx = 15)
        
        self.choose_label_label = tk.Label(self.widgets_frame, text = "Label:", font = "Arial 10 bold")
        self.choose_label_label.pack(side = "top", anchor = tk.NW, padx = 15)
        
        self.label_drop = ttk.Combobox(self.widgets_frame, values = self.labels)
        self.label_drop.set(["-"])
        self.label_drop.pack(side = "top", anchor = tk.N, padx = 15)
        
        divider2 = tk.Label(self.widgets_frame, text = "________________", font = "Arial 8")
        divider2.pack(side = "top", pady = 5)
        
        self.label_csv_info = tk.Label(self.widgets_frame, text = "n.a.", font = "Arial 8 italic")
        self.label_csv_info.pack(side = "bottom", anchor = tk.NW, padx = 15)
        
        self.label_csv = tk.Label(self.widgets_frame, text = "CSV File:", font = "Arial 8")
        self.label_csv.pack(side = "bottom", anchor = tk.NW, padx = 15)

        self.label_group = tk.Label(self.widgets_frame, text = "Group: n.a.", font = "Arial 8")
        self.label_group.pack(side = "bottom", anchor = tk.NW, padx = 15)
        
        self.label_label = tk.Label(self.widgets_frame, text = "Molecule: n.a.", font = "Arial 8")
        self.label_label.pack(side = "bottom", anchor = tk.NW, padx = 15)
        
        self.label_information = tk.Label(self.widgets_frame, text = "Information on current data:", font = "Arial 8 bold")
        self.label_information.pack(side = "bottom", anchor = tk.NW, padx = 15)
                    
    def create_canvas(self):
        self.console_height = 250
        self.master.update()
        self.left_frame_height = int(self.master.winfo_height() - self.console_height)
        self.left_frame_width = int((self.master.winfo_width() - self.widget_width) / 2)
        self.panel1 = tk.PanedWindow(self.window_frame, orient = "vertical", relief = tk.GROOVE)
        self.panel2 = tk.PanedWindow(self.window_frame, orient = "horizontal", relief = tk.GROOVE)
        self.panel2.pack(fill = "both", expand = 1)
        self.panel1.add(self.panel2)
        self.left_frame = tk.Frame(self.panel2, relief = tk.GROOVE, width = self.left_frame_width, height = self.left_frame_height)
        self.left_frame.pack(side = "left", anchor = tk.NW, fill = "both", expand = 1)
        self.left_frame.pack_propagate(0)
        self.left_frame_label = tk.Label(self.left_frame, text = "Image", relief = tk.GROOVE)
        self.left_frame_label.pack(expand = 1, fill = "both")
        self.right_frame = tk.Frame(self.panel2, relief = tk.GROOVE)
        self.right_frame.pack(fill = "both", expand = 1)
        self.right_frame_label = tk.Label(self.right_frame, text = "Plot", relief = tk.GROOVE)
        self.right_frame_label.pack(fill = "both", expand = 1)
        self.panel2.add(self.left_frame)
        self.panel2.add(self.right_frame)
        self.console_frame = tk.Frame(self.panel1, relief = tk.GROOVE, borderwidth = 1)
        self.console_frame.pack(side = "right", fill = "both", expand = 1)
        
        self.console_output("Welcome to SpecApp :)")
        self.panel1.add(self.console_frame)
        self.panel1.pack(fill = "both", expand = 1)
        
    def get_file(self):
        self.file = fd.askopenfile(title = "Load Image", mode = "r", filetypes =[('Images', '*.png')]).name
        self.label = self.file.split("/")[-1].split(".")[0]
        self.group = self.file.split("/")[-2]
        self.label_group.config(text = f"Group: {self.group}")
        self.label_label.config(text = f"Molecule: {self.label}")
        
        if self.file is not None:
            img = Image.open(self.file)
            self.img = ImageTk.PhotoImage(img)
        
        self.set_image_frame()
        self.console_output(f"Successfully loaded {self.label} of the {self.group} group.")
    
    def delete_children(self, widget):
        try:
            for children in widget.winfo_children():
                children.destroy()
        except:
            self.console_output("Ups, something went wrong with the destruction.")
    
    def set_image_frame(self):
        try:
            self.left_frame_label.destroy()
            self.delete_children(self.left_frame) 
        except:
            self.delete_children(self.left_frame)    
        
        self.image = ScrollableImage(self.left_frame, image = self.img)
        
    def convert_image(self):
        self.console_output("Please wait a second...")
        self.master.update()
        try:
            self.plot = i2p.extract_SDBS_data(self.file)
        except:
            self.console_output("Could not convert anything. Possible errors: no data given or no plot found.")
        
        self.delete_children(self.right_frame)
        self.plot_canvas = PlotImage(self.x, self.plot, master = self.right_frame)
        self.console_output("Successfully converted the image to a plot.")

    def fft(self):
        if self._switched == False:
            self.switch_positions()
        self.fft = np.fft.fft(self.plot).real
        y = self.fft[0 : int(len(self.fft) / 2)]
        x = np.arange(len(self.fft) / 2)
        self.delete_children(self.left_frame)
        self.fft_canvas = PlotImage(x, y, master = self.left_frame)
        self.panel2.add(self.left_frame)
        self.panel2.add(self.right_frame)
    
    def new_file(self):
        return
        
    def switch_positions(self):
        self.left_frame.pack_forget()
        self.right_frame.pack_forget()
        if self._switched == False:
            self.right_frame.pack(side = "left", anchor = tk.NW, fill = "both", expand = 1)
            self.right_frame.configure(width = self.left_frame_width, height = self.left_frame_height)
            self.left_frame.configure(width = None, height = None)
            self.right_frame.pack_propagate(0)
            self.left_frame.pack(fill = "both", expand = 1)
            self._switched = True
            self.panel2.add(self.right_frame)
            self.panel2.add(self.left_frame)
        elif self._switched == True:
            self.left_frame.pack(side = "left", anchor = tk.NW, fill = "both", expand = 1)
            self.left_frame.pack_propagate(0)
            self.left_frame.configure(width = self.left_frame_width, height = self.left_frame_height)
            self.right_frame.configure(width = None, height = None)
            self.right_frame.pack(fill = "both", expand = 1)
            self._switched = False
            self.panel2.add(self.left_frame)
            self.panel2.add(self.right_frame)        
        self.console_output("Switched Frames.")
    
    def water_disturbance(self):
        try:
            csv_filename = self.csv_path.split("/")[-1].split(".")[0]
            spectrum = ManipulateSpectrum(self.label, self.group, csv_filename)
            new_window = tk.Toplevel()
            disturbed = spectrum.water_disturbance(0.5)
            self.delete_children(self.left_frame)
            self.disturbed_plot = PlotImage(self.x, disturbed, master = self.left_frame) 
        except:
            self.console_output("Something went wrong.")
    
    def load_csv(self):
        self.csv_path = fd.askopenfile(title = "Set CSV File", mode = "r", filetypes =[('CSV', '*.csv')]).name
        self.label_csv_info.config(text = f"{'/'.join(self.csv_path.split('/')[4:])}")
        self.console_output(f"Loaded .csv-file: {self.label_csv_info['text']}")
        try:
            self.get_csv_groups()
        except:
            pass
        
    def save_csv(self):
        self.csv_path = fd.asksaveasfile(title = "Create new CSV file", filetypes = [('CSV', '*.csv')], defaultextension = "*.csv").name
        self.label_csv_info.config(text = f"{'/'.join(self.csv_path.split('/')[4:])}")
        self.console_output(f"Saved .csv-file at {self.label_csv_info['Text']}")
    
    def load_plot(self):
        try:
            self.delete_children(self.right_frame)
            self.label = self.label_drop.get()
            self.group = self.group_drop.get()
            mask = self.df['label'] == self.label
            self.plot = self.df[mask].values[0, 2:]
            self.plot_canvas = PlotImage(self.x, self.plot, master = self.right_frame)
            self.console_output(f"Loaded {self.label} from .csv-file.")
            self.label_group.config(text = f"Group: {self.group}")
            self.label_label.config(text = f"Molecule: {self.label}")
        except:
            self.console_output("Please specify correct label and group first.")        
    
    def ask_load_save(self):
        answer = messagebox.askyesnocancel("CSV File", "Do you want to load an existing file?")
        if answer == True:
            self.load_csv()
        elif answer == False:
            self.save_csv()
        else:
            pass            
        
    def save_plot(self):
        if not self.csv_path:
            self.ask_load_save()
        
        data = np.array([self.label, self.group] + self.plot)
        data = data.reshape([1, len(data)])
        cols = ["label", "group"] + [str(self.x[i]) for i in range(len(self.x))]
        try:
            old_df = pd.read_csv(self.csv_path)
            if (self.label in old_df.values) and (old_df.loc[old_df["label"] == self.label]["group"].values[0] == self.group):
                    self.console_output(f"Molecule '{self.label}' is already part of the CSV-file.")
                    df = old_df
            else:
                new_df = pd.DataFrame(data, columns = cols)
                df = pd.concat([new_df, old_df], ignore_index = True)
                self.console_output(f"Molecule '{self.label}' is added to the CSV-file.")
      
        except:
            df = pd.DataFrame(data, columns = cols)
            self.console_output(f"A new .csv-file with {self.label} is stored at: {self.csv_path}")        
                
        df.to_csv(self.csv_path, index = False)
        self.get_csv_groups()
    
    def console_output(self, string):
        if len(self.output) == 0:
            self.output.append(string)
            self.console_text = tk.Text(self.console_frame, fg = "black", font = "courier 10 bold",  pady = 10, padx = 15, relief = tk.GROOVE)
            self.text_scrolly = tk.Scrollbar(self.console_frame, orient = "vertical", command = self.console_text.yview)
            self.text_scrolly.pack(side = "right", anchor = tk.NW, expand = 0, fill = "y")
            self.text_scrolly.pack_propagate(0)
            self.console_text.pack(side = "top", anchor = tk. SW, fill = "both", expand = 1)
            self.console_text.insert("end", f"{self.output[-1]}\n")
        else:
            self.console_text.configure(font = "courier 10")
            self.output.append(string)
            self.console_text.insert("end", f"{self.output[-1]}\n")
        self.console_text.see(tk.END)
    
def start():
    root = tk.Tk()
    root.wm_title("Spectral Analysis Tool")
    root.iconbitmap(os.path.dirname(__file__) + '/icons/icon.png')
    root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file = os.path.dirname(__file__) + '/icons/icon.png'))
    root.configure(width = 1400, height = 700)
    root.geometry("1400x700")
    app = SpecApp(root)
    app.mainloop()
