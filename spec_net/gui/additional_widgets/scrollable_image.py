import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ScrollableImage(tk.Frame):
    def __init__(self, master=None, **kw):
        self.img = kw.pop('image', None)
        sw = kw.pop('scrollbarwidth', 15)
        super(ScrollableImage, self).__init__(master=master, **kw)
        self.N_frame = tk.Frame(master, width = int(master.winfo_width()), height =  int(master.winfo_height() - sw))
        self.S_frame = tk.Frame(master, width = int(master.winfo_width() - sw), height = sw)
        self.S_frame.pack(side = "bottom", anchor = tk.SW, expand = 0, fill = "x")
        self.S_frame.pack_propagate(0)
        self.N_frame.pack(side = "top", anchor = tk.NW, expand = 1, fill = "both")
        self.image = tk.Canvas(self.N_frame, relief = tk.GROOVE, width = int(master.winfo_width() - sw), height =  int(master.winfo_height() - sw))
        self.image.create_image(0, 0, anchor = tk.NW, image = self.img)
        self.filler = tk.Frame(self.S_frame, relief = tk.GROOVE, width = 17, height = sw)
        self.filler.pack(side = "right", anchor = tk.SE)
        self.filler.pack_propagate(0)
        self.scrollx = tk.Scrollbar(self.S_frame, orient = "horizontal", command = self.image.xview, width = self.image.winfo_width())
        self.scrollx.pack(side = "left", anchor = tk.SW, fill = "x", expand = 1)
        self.scrollx.pack_propagate(0)
        self.scrolly = tk.Scrollbar(self.N_frame, orient = "vertical", command = self.image.yview)
        self.scrolly.pack(side = "right", anchor = tk.NW, expand = 0, fill = "y")
        self.scrolly.pack_propagate(0)
        self.image.pack(side = "right", anchor = tk.NW, expand = 1, fill = "both")
        self.image.update()
        self.scrollx.configure(width = self.image.winfo_width())
        self.image.configure(yscrollcommand=self.scrolly.set, xscrollcommand = self.scrollx.set)        
        self.image.config(scrollregion = self.image.bbox("all"))

class PlotImage(tk.Frame):
    def __init__(self, x, y, master = None, **kw):
        super().__init__(master = master, **kw)
        fig = Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, y)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 100)
        ax.grid(True)
        self.canvas = FigureCanvasTkAgg(fig, master = master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side = "top", anchor = tk.NW, expand = 1, fill = "both")