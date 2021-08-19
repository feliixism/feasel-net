import tkinter as tk
import tkinter.filedialog as fd
import pandas as pd
from .additional_widgets.scrollable_image import ScrollableImage
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..preprocess import image_plot_conversion as i2p
from ..preprocess.datahandling import CSVFile
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# class Start(tk.Frame):
#     def __init__(self, master = None):
#         super().__init__(master)
#         self.master = master
#         self.pack()
#         self.create_widgets()
#         self.create_canvas()
#         self.create_menu()

class StartPage(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_frame()
        self.create_widgets()
        self.create_canvas()
        self.create_menu()
        
    def create_menu(self):
        menu = tk.Menu(self.master, tearoff=False)
        self.master.config(menu = menu)
        
        fileMenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label = "File", menu = fileMenu)
        fileMenu.add_command(label = "New File", command = self.create_file_frame)
        fileMenu.add_command(label = "Load Image", command = self.get_file)
        fileMenu.add_command(label = "Load CSV File", command = self.load_csv)
        fileMenu.add_command(label = "Save Plot", command = self.save_plot)
        fileMenu.add_separator()
        fileMenu.add_command(label = "Exit", command = self.master.destroy)
        
        editMenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label = "Edit", menu = editMenu)
        editMenu.add_command(label = "FFT")
        editMenu.add_command(label = "Mask")
    
    def create_frame(self):
        self.frame = tk.Frame(self.master, bg = "white", relief = tk.GROOVE)
        self.frame.pack(side = "top", anchor = tk.NW, fill = "both", expand = True, padx = 5, pady = 5)

    def create_widgets(self):
        
        self.widgets_frame = tk.Frame(self.frame, relief = tk.GROOVE, borderwidth = 1, width = 200)
        self.widgets_frame.pack(side = "right", anchor = tk.NE, fill = "y")
        self.widgets_frame.pack_propagate(0)
        
        self.convert = tk.Button(self.widgets_frame, text = "Convert", width = 10, height = 1)
        self.convert["command"] = self.convert_image
        self.convert.pack(side = "top", padx = 5, pady = 2)
        
        self.save = tk.Button(self.widgets_frame, text = "Save Plot", width = 10, height = 1, command = self.save_plot)
        self.save.pack(side = "top", padx = 5, pady = 2)

        self.quit = tk.Button(self.widgets_frame, text="Quit!", fg="red",
                              command=self.master.destroy, width = 10, height = 1)
        self.quit.pack(side = "top", padx = 5, pady = 2)
        
        self.label_csv_info = tk.Label(self.widgets_frame, text = "-", font = "Arial 8 italic")
        self.label_csv_info.pack(side = "bottom", anchor = tk.NW)
        
        self.label_csv = tk.Label(self.widgets_frame, text = "Current CSV File:", font = "Arial 8")
        self.label_csv.pack(side = "bottom", anchor = tk.NW)
                    
    def create_canvas(self):
        self.top_frame = tk.Frame(self.frame, height = int(3/4 * self.master.winfo_reqheight()), relief = tk.GROOVE, borderwidth = 1)
        self.top_frame.pack(side = "top", anchor = tk.NW, fill = "both", expand = True)
        self.top_frame.pack_propagate(0)
        
        self.lower_frame = tk.Frame(self.frame, height = int(1/4 * self.master.winfo_reqheight()), relief = tk.GROOVE, borderwidth = 1)
        self.lower_frame.pack(side = "bottom", anchor = tk.NW, fill = "both", expand = True)
        self.lower_frame.pack_propagate(0)
        
    def create_file_frame(self):
        try:
            for child in self.file_frame.winfo_children():
                child.forget()
            self.file_frame.forget()
            print("Destroyed all")
        except:
            self.file_frame = tk.Frame(self.top_frame, width = self.top_frame.winfo_width(), height = self.top_frame.winfo_height(), relief = tk.GROOVE, borderwidth = 1)
        self.file_frame.pack(side = "top", anchor = tk.NW, fill = "both", expand = True)
        
    def get_file(self):
        self.create_file_frame()
        self.file = fd.askopenfile(title = "Load Image", mode = "r", filetypes =[('Images', '*.png'), ('Plots', '*.npy')]).name
        self.label = self.file.split("/")[-1].split(".")[0]
        self.group = self.file.split("/")[-2]
        
        if self.file is not None:
            img = Image.open(self.file)
            self.img = ImageTk.PhotoImage(img)
        
        self.set_image_frame(self.file_frame.winfo_width(), self.file_frame.winfo_height())
    
    def set_image_frame(self, width, height):
        self.image_frame = tk.Frame(self.file_frame, width = width, height = height)
        self.image_frame.pack(side = "left", anchor = tk.NW, fill = "both",  expand = True)
        self.image_canvas = tk.Canvas(self.image_frame, width = width - 25, height = height - 25, relief = tk.GROOVE, borderwidth = 1)
        self.image_canvas.create_image(0, 0, anchor = tk.NW, image = self.img)
        self.image_canvas.grid(row=0, column=0)
        self.image_scrollx = tk.Scrollbar(self.image_frame, orient = "horizontal", command = self.image_canvas.xview)
        self.image_scrollx.grid(row=1, column=0, sticky="ew")
        self.image_scrolly = tk.Scrollbar(self.image_frame, orient = "vertical", command = self.image_canvas.yview)
        self.image_scrolly.grid(row=0, column=1, sticky="ns")
        
        self.image_canvas.configure(yscrollcommand=self.image_scrolly.set, xscrollcommand=self.image_scrollx.set)
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

    def convert_image(self):
        try:
            self.image_frame.destroy()
            self.set_image_frame(int(self.file_frame.winfo_width() / 2), self.file_frame.winfo_height())
        except:
            print("No file to convert")
        self.plot = i2p.extract_SDBS_data(self.file)
        self.x = np.round(np.linspace(4000, 400, 3600), 0)
        fig = Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.x, self.plot)
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(0, 100)
        
        self.plot_frame = tk.Frame(self.file_frame, relief = tk.GROOVE, borderwidth = 1, width = int(self.file_frame.winfo_width() / 2), height = self.file_frame.winfo_height())
        self.plot_frame.pack(side = "right", anchor = tk.NW, expand = True, fill = "both")
        
        self.plot_canvas = FigureCanvasTkAgg(fig, master = self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side = "top", anchor = tk.NW)
    
    def load_csv(self):
        self.csv_path = fd.askopenfile(title = "Set CSV File", mode = "r", filetypes =[('CSV', '*.csv')]).name
        self.label_csv_info.config(text = f"{'/'.join(self.csv_path.split('/')[4:])}")
    
    def save_plot(self):
        try:
            print(f"CSV-Path: {self.csv_path}")
        except:
            self.csv_path = fd.askopenfile(title = "Set CSV File", mode = "r", filetypes =[('CSV', '*.csv')]).name
            self.label_csv_info.config(text = f"{'/'.join(self.csv_path.split('/')[4:])}")
            print(f"CSV-Path: {self.csv_path}")
        try:
            data = np.array([self.label, self.group] + self.plot)
            data = data.reshape([1, len(data)])
            old_df = pd.read_csv(self.csv_path)
            cols = ["label", "group"] + [str(self.x[i]) for i in range(len(self.x))]
            if self.label not in old_df.values:
                new_data = pd.DataFrame(data, columns = cols)
                df = pd.concat([new_data, old_df], ignore_index = True)
                print(f"{self.label} is added to the CSV-file.")
            else:
                print(f"{self.label} is already part of the CSV-file.")
                df = old_df
        
        except:
            df = pd.DataFrame(data, columns = cols)
            print(f"A new CSV-file with {self.label} is stored at: {self.csv_path}")
        
        df.to_csv(self.csv_path, index = False)
    
def start():
    root = tk.Tk()
    root.wm_title("Spectral Analysis Tool")
    root.iconbitmap(os.path.dirname(__file__) + '/icons/icon.png')
    root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file = os.path.dirname(__file__) + '/icons/icon.png'))
    root.configure(width = 1400, height = 700)
    root.geometry("1400x700")
    app = StartPage(root)
    app.mainloop()
