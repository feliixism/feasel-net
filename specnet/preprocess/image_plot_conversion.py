#function to extract spectral data from SDBS database
import numpy as np
from PIL import Image
import math

def binary_image(img):
    if len(img.shape) == 3:
        img = np.mean(img, axis = -1)
    threshold = np.amax(img) / 2
    mask = img >= threshold
    return mask.astype(int)

def stretch_plot(plot, interval = 3600):
    """
    Stretches each plot to specified length.

    Parameters
    ----------
    plot : np-array (1D: float)
        Plot with original length.
    interval : int, optional
        Length of the plot after stretch. The default is 3600.

    Returns
    -------
    stretched_plot : np-array (1D: float)
        Plot after being stretched.

    """

    delta_x = len(plot) / interval
    stretched_plot = []
    
    for i in range(0, interval):
        x = delta_x * i
        x_floor = math.floor(x)
        x_ceil = math.ceil(x)
        ratio = (x - x_floor)
        
        try:
            y = plot[x_floor] + (plot[x_ceil] - plot[x_floor]) * ratio
        except:
            y = plot[x_floor] + (plot[x_ceil - 1] - plot[x_floor - 1]) * ratio
        
        stretched_plot.append(y)
    
    return stretched_plot

def search_longest_lines(img, direction = "horizontal", iterations = 2):
    """
    Searches for the longest lines in the specified direction.
    Hereby, a line is defined as a continous sequence of grayscale values '1'.

    Parameters
    ----------
    img : np-array (2D: float)
        The image that is searched for longest lines.
    direction : str, optional
        Either 'horizontal' or 'vertical' are possible directions. The default is "horizontal".
    iterations : int, optional
        Determines the number of longest lines. The default is 2.

    Returns
    -------
    lines : np-array (2D: float)
        List of the longest lines.

    """
    col_bound = img.shape[1]
    row_bound = img.shape[0]
    
    lines =[]
    
    while True:
        length = 0
        
        line = [[0, 0], [0, 0]]
        
        if direction == "horizontal":
            for line in lines:
                img[line[0][0]] = 1
            for row in range(0, row_bound): #rows
                for col in range(0, col_bound): #cols
                    
                    if (img[row, col] == 0) and (col_bound > length + 1):
                        line_length = 0
                        
                        while (img[row, col + line_length] == 0) and (col + line_length + 1 < col_bound):
                            line_length += 1
                        
                        if line_length >= length:
                            length = line_length
                            line = [[row, col], [row, col + length]]
        
        elif direction == "vertical":
            for line in lines:
                img[ : , line[0][1]] = 1
            for col in range(0, col_bound):
                for row in range(0, row_bound):
                    
                    if (img[row, col] == 0) and (row_bound > length + 1):
                        line_length = 0
                        
                        while (img[row + line_length, col] == 0) and (row + line_length + 1 < row_bound):
                            line_length += 1
                        
                        if line_length >= length:
                            length = line_length
                            line = [[row, col], [row + length, col]]
                        
        lines.append(line)
        
        if len(lines) > iterations - 1:
            break
    
    return lines

def find_plot_coordinates(img):
    """
    Searches for the intersections of the longest lines calculated by search_longest_lines().

    Parameters
    ----------
    img : np-array (2D: float)
        Image that is being searched for.

    Returns
    -------
    x_1 : int
        x-position of the upper left corner of the biggest rectangle.
    x_2 : int
        x-position of the lower right corner of the biggest rectangle.
    y_1 : int
        y-position of the upper left corner of the biggest rectangle.
    y_2 : int
        y-position of the lower right corner of the biggest rectangle.

    """
    horizontal_lines = search_longest_lines(img, direction = "horizontal")
    vertical_lines = search_longest_lines(img, direction = "vertical")
    
    coordinates = []
    for m in range(0, len(horizontal_lines)):  
        row = horizontal_lines[m][0][0]
        horizontal = []
        for i in range(horizontal_lines[m][0][1] - 5, horizontal_lines[m][1][1] + 5):
            horizontal.append(f"{row},{i}")
        horizontal = np.array(horizontal)
        for n in range(0, len(vertical_lines)):
            col = vertical_lines[n][0][1]
            vertical = []
            for j in range(vertical_lines[n][0][0] - 5, vertical_lines[n][1][0] + 5):
                vertical.append(f"{j},{col}")
            vertical = np.array(vertical)
            
            for horizontalCoordinate in range(0, len(horizontal)):
                for verticalCoordinate in range(0, len(vertical)):
                    if horizontal[horizontalCoordinate] == vertical[verticalCoordinate]:
                        coordinates.append([horizontal[horizontalCoordinate].split(",")[0], horizontal[horizontalCoordinate].split(",")[1]])
    
    coordinates = np.array(coordinates).astype(int)
    
    x_1 = np.amin(coordinates[ : , 1]) + 1
    x_2 = np.amax(coordinates[ : , 1]) - 1
    y_1 = np.amin(coordinates[ : , 0]) + 1
    y_2 = np.amax(coordinates[ : , 0]) - 1
    
    return x_1, x_2, y_1, y_2

def extract_SDBS_data(path):
    """
    Converts image data to plots of spectra.

    Parameters
    ----------
    path : str
        Path of image that is being converted.

    Returns
    -------
    plot : np-array (1D: float)
        Spectral plot.

    """
    img = Image.open(path)
    img = np.array(img)
    
    img = binary_image(img)
    
    #clippingPoints
    x_1, x_2, y_1, y_2 = find_plot_coordinates(img)
    
    clipped_image = img[y_1 : y_2, x_1 : x_2]

    # clipping images
    ratio_MIR_plot = 20 / 52
    
    img_1 = clipped_image[ : , 0 : int(ratio_MIR_plot * clipped_image.shape[1])]
    img_2 = clipped_image[ : , int(ratio_MIR_plot * clipped_image.shape[1]) : clipped_image.shape[1]]    
    
    # generating first half of plot
    plot_1 = []
    for i in range(0, img_1.shape[1]):
        bar_length = 1
        for j in range(0, img_1.shape[0]):
            if img_1[j, i] == 0:
                bar_length += 1
                y = 100 - (j - bar_length / 2) / img_1.shape[0] * 100
            else:
                if bar_length > 1:
                    break
        plot_1.append(y)
    
    augmented_plot_1 = []
    for i in range(0, len(plot_1) - 1):
        augmented_plot_1.append(plot_1[i])
        augmented_plot_1.append((plot_1[i] + plot_1[i + 1]) / 2)
    
    plot_2 = []
    for i in range(0, img_2.shape[1]):
        bar_length = 1
        for j in range(0, img_2.shape[0]):
            if img_2[j, i] == 0:
                bar_length += 1
                y = 100 - (j - bar_length / 2) / img_2.shape[0] * 100
            else:
                if bar_length > 1:
                    break
        plot_2.append(y)
    
    plot = augmented_plot_1 + plot_2
    plot = stretch_plot(plot)
    
    return plot