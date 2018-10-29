from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def showSlice2D(image, sketchlistslice):
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi= 80, facecolor='w', edgecolor='k')
    rows, cols = image.shape
    image = ax.imshow(image[:, :])
    
    #ax.remove()
    #for slice in sketchlistslice:
        #ax.add_artist(sketchlistslice)

def showSlice3D(X, currentslice, sketchlistslice):
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi= 80, facecolor='w', edgecolor='k')
    slices, rows, cols = X.shape
    
    image = ax.imshow(X[currentslice, :, :])
    ax.remove()
    
    for sketch in sketchlistslice:
        ax.add_artist(sketch)
        
def add_circle2D(x, y, radius, sketch_list):
    
    sketch_list.append(plt.Circle((x,y), radius, fill=False, edgecolor ='r' ))
    return sketch_list

def add_circle3D(x, y, radius, slice, sketch_list):
    if slice in sketch_list:
        sketch_list[slice].append(plt.Circle((x,y), radius, fill=False, edgecolor ='r' ))
    else:
        sketch_list[slice] = [plt.Circle((x,y),  radius, fill=False, edgecolor ='r' )]
    return sketch_list

def add_blobs2D(x, y, scale, sketch_list):
    radius=scale*np.sqrt(2)
    sketch_list=add_circle2D(x, y, radius, sketch_list)
    return sketch_list

def add_blobs3D(x, y, scale, slice,  z_x, sketch_list):
    radius=scale*np.sqrt(3)
    sketch_list=add_circle3D(x, y, radius, slice, sketch_list)
    step = 0
    
    while (radius**2 - (step*z_x)**2 > 0.0):
        R = np.sqrt(radius**2 - (step*z_x)**2)
        sketch_list=add_circle3D(x, y, R, slice-step, sketch_list)
        sketch_list=add_circle3D(x, y, R, slice+step, sketch_list)
        step=step+1
    return sketch_list