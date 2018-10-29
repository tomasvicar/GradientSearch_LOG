import os
from pathlib import Path
from Functions import blob
from scipy import spatial
from numpy import zeros, logspace, mean, log10, argsort, shape, arccos, asarray, linspace, pi, digitize
from numpy.linalg import norm
import pandas as pd
import numpy as np
import collections
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from matplotlib.patches import Ellipse


def sigma2abc(sx,sy,t):

    a =  np.cos(t)**2 / (2 *sx **2) +  np.sin(t)**2 / (2 * sy ** 2);
    b = - np.sin(2 * t)/ (4 * sx ** 2) + np.sin(2 * t) / (4 * sy ** 2);
    c =  np.sin(t) ** 2 / (2 * sx **2) + np.cos(t) ** 2 / (2 * sy**2);
    
    
    return a,b,c

def abc2sigma(a,b,c):

    x=((a+c)+np.sqrt((a-c)**2+4*b**2))/2;
    y=a+c-x;
    t=np.arccos((a-c)/(x-y))/2;
    
    sx=np.sqrt(1/(2*x));
    sy=np.sqrt(1/(2*y));
    
    return sx,sy,t


def draw_elipses(img,x,y,sx,sy,t,abc=False):
    """
    input can be img,x,y,a,b,c if abc=True
    t - radians
    """
    
    if abc:
        sx,sy,t=abc2sigma(sx,sy,t)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi= 80, facecolor='w', edgecolor='k')
    rows, cols = data.shape
    ax.imshow(img)
    
    for x_tmp,y_tmp,sx_tmp,sy_tmp,t_tmp in zip(x,y,sx,sy,t):
            ax.add_artist(Ellipse((y_tmp,x_tmp), sx_tmp*np.sqrt(2),sy_tmp*np.sqrt(2),t_tmp*180/np.pi, fill=False, edgecolor ='r' ))


def isDir(address):
    if (os.path.isdir(address)):
        print("")
    elif (os.path.isdir(Path(address).parents[0])):
        os.mkdir(address)
    else:
        os.mkdir(Path(address).parents[0])
        os.mkdir(address)
        
def getDataAndHeader(datapath, filename):
    data=pd.read_csv(datapath+filename, header=None, dtype=np.float32, skiprows=10, delimiter= '\t')
    header=pd.read_csv(datapath+filename, header=None, nrows=10, delimiter= '\t')
    data=np.asarray(data)
    header=np.asarray(header)
    headerdictionary={'mode':header[0, 1], 'dataname': header[2,1], 'datatype': header[3, 1], 'datamodality': header[4, 1], 'radiusmin' : header[5, 1], 'radiusmax' : header[6, 1], 'z_x' : float(header[7, 1]), 'y_x' : float(header[8, 1]), 'brightondark': header[9,1]}
    return data, headerdictionary
        
def computeDistance(dataslice):
    euclidDist_B2B = norm(dataslice[:, None, :] - dataslice[None, :, :], axis=2)
    meanDist_B2B=mean(euclidDist_B2B, axis=1)
    scalenormalizedDist_B2B=euclidDist_B2B/meanDist_B2B[:, None]
    #logrDist_B2B=log10(euclidDist_B2B/meanDist_B2B[:, None])
    return euclidDist_B2B, scalenormalizedDist_B2B

def computePlaneCoefficients(euclidDist_B2B, dataslice):
    planecoefficients=[]
    for row in euclidDist_B2B:
        k = 4 # 4 nearest neighbours
        index= argsort(row)
        planecoefficients.append(blob.getTangentialPlane(dataslice[index[1:k], :]))
    return planecoefficients
    dict.fromkeys(range(1, 61))

def computeTheta(planecoefficients, dataslice):
    theta=zeros((shape(dataslice)[0], shape(dataslice)[0]))
       
    for i in range(0, shape(dataslice)[0]):
        for j in range(0, shape(dataslice)[0]):
            
            zd=dataslice[i,0]-dataslice[j,0]
            yd=dataslice[i,1]-dataslice[j,1]
            xd=dataslice[i,2]-dataslice[j,2]
            
            theta[i,j]=arccos(1-spatial.distance.cosine([xd,yd,zd], asarray(planecoefficients[i])))
            
    return theta

def computeCombinedBinIndex(scalenormalizedDist_B2B, theta):
    
    r_inner=0.125
    r_outer=2
    n_rbins=5
    n_thetabins=12
    r_bin_edges=logspace(log10(r_inner), log10(r_outer), n_rbins);
    r_bin_edges=r_bin_edges.tolist()
    r_bin_edges.insert(0,0)
    r_bin_edges=asarray(r_bin_edges)
    theta_bin_edges = linspace(0, 2*pi, n_thetabins+1);
    
    r_bin_index=zeros((shape(scalenormalizedDist_B2B)[0], shape(scalenormalizedDist_B2B)[1]))
    theta_bin_index=zeros((shape(scalenormalizedDist_B2B)[0], shape(scalenormalizedDist_B2B)[1]))
    r_bin_index=digitize(scalenormalizedDist_B2B, r_bin_edges)
    theta_bin_index=digitize(theta, theta_bin_edges)
    combined_bin_index=(r_bin_index-1)*n_thetabins+theta_bin_index-1
    return combined_bin_index

def computeDescriptor(combined_bin_index):
    descriptor=zeros((shape(combined_bin_index)[0], 60))
    for row in range(0, shape(combined_bin_index)[0]):
        for bin in range(0, 60):
            descriptor[row, bin]=combined_bin_index[row, :].tolist().count(bin)
            
    return descriptor     


def evaluateMetricsStandard(predictions, groundtruth, radius=radiusmatrix):
    distancematrix=distance_matrix(predictions, groundtruth)
    rowindex, colindex=linear_sum_assignment(distancematrix)
    distancevalue=distancematrix[rowindex, colindex]
    rowindex, colindex=rowindex[distancevalue<radius], colindex[distancevalue<radius]
    TP=len(rowindex)
    FP=np.shape(predictions)[0] -TP
    FN=np.shape(groundtruth)[0] -TP
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    jaccardindex=TP/(TP+FP+FN)
    F1score=(2*TP)/(2*TP+FP+FN)
    return TP, FP, FN, recall, precision, jaccardindex, F1score 
    

    
    
    
    
