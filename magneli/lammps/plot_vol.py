import numpy as np

from mayavi import mlab
import mayavi

import h5py
import os

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"

# --------------------------------------------------------------------------------------------------

def plot_vol(file_name='293K_quenched.hdf5',scale=1.0):

    a = 2*np.pi/4.611; c = 2*np.pi/2.977

    with h5py.File(file_name,'r') as db:
        h = db['H'][...]; k = db['K'][...]; l = db['L'][...]
        signal = db['sq_elastic'][...]

    h *= a; k *= a; l *= c

    signal = np.nan_to_num(signal,nan=0.0,posinf=0.0,neginf=0.0)*scale

    fig = mlab.figure(1, bgcolor=(1,1,1), fgcolor=(0,0,0), size=(500, 500))
    mlab.clf()

    extent = [h.min(),h.max(),k.min(),k.max(),l.min(),l.max()]
    h, k, l = np.meshgrid(h,k,l,indexing='ij')

    contours = []
    for ii in np.linspace(0.15,0.3,250): 
        contours.append(ii)
    mlab.contour3d(h,k,l,signal,contours=contours,color=(1,0.5,1),  #color=(1,0.5,1),
            transparent=True,opacity=0.005,figure=fig)

    contours = []
    for ii in np.linspace(0.25,0.5,250):
        contours.append(ii)
    mlab.contour3d(h,k,l,signal,contours=contours,color=(1,0.75,0),
            transparent=True,opacity=0.01,figure=fig)

    if signal.max() >= 10:
        vmax = 9
    else:
        vmax = signal.max()*0.9
    contours = []
    for ii in np.linspace(0.5,vmax,100):
        contours.append(ii)
    mlab.contour3d(h,k,l,signal,contours=contours,color=(0.5,0.5,0), #color=(1,0.75,0)
            transparent=True,opacity=1.0,figure=fig)

    mlab.outline(color=(0,0,0),line_width=1,extent=extent)
    mlab.orientation_axes()

    fig.scene.parallel_projection = True

    center = np.array([h.mean(),k.mean(),l.mean()])
    views = {'iso':(45.0, 54.735610317245346, 'auto', center),
              'X':(180.0, 90.0, 'auto', center), 
              'Y':(-90.0, 90.0, 'auto', center), 
              'Z':(0.0, 180.0, 'auto', center)}
              #'a':(72.28808017764584, 78.43057609125368, 3.346065214951226*zoom, center),
              #'b':(34.43222890423177, 165.83420283990634, 3.346065214951206*zoom, center),
              #'c':(171.9282308683452, 101.1974312536663, 3.3460652149512016*zoom, center)}

    cam = fig.scene.camera
    cam.parallel_scale = 1

    #view = views[key]
    #mlab.view(*view)

    #mlab.savefig(fig_name,size=(1000,1000),magnification=1.0)

    mlab.show()

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    file_name = 'psf_STRUFACS.hdf5'
    plot_vol(file_name,scale=1/50)








