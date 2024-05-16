
import matplotlib.pyplot as plt
from matplotlib import cm # set colormap and mask nans etc
from matplotlib import colors # normalize colorscale 
from matplotlib.widgets import CheckButtons, TextBox # widgets
import numpy as np
import h5py

# --------------------------------------------------------------------------------------------------

# read signal and error from my hdf5 file
f_300K = 'no_couple_6k0_300K.hdf5'
with h5py.File(f_300K,'r') as db:
    H = db['Dim_0'][...]
    E = db['Dim_1'][...]
    signal = db['signal'][...]
    error = db['error'][...]

# --------------------------------------------------------------------------------------------------

# sets the initial color map scale
vmax = np.nanmax(signal)*0.01
vmin = np.nanmin(signal)

# initial colormap is linear
norm = colors.Normalize(vmin,vmax)

# this changes the color of nan's to whatever you want
cmap = cm.viridis # the colormap
cmap.set_extremes(bad='w',under=None,over=None) 

# figure and axes handles
fig, ax = plt.subplots(1,2,figsize=(12,5))

# pad around colormaps to put textbox/checkbox widgets
fig.subplots_adjust(bottom=0.2,top=0.9)

# define and place the widgets
minbox_ax = fig.add_axes([0.7,0.025,0.2,0.05]) 
minbox = TextBox(minbox_ax,label='vmin',initial=vmin) # vmin textbox
maxbox_ax = fig.add_axes([0.7,0.075,0.2,0.05])
maxbox = TextBox(maxbox_ax,label='vmax',initial=vmax) # vmax textbox
logbox_ax = fig.add_axes([0.75,0.95,0.1,0.05])
logbox = CheckButtons(logbox_ax,labels=['log']) # log/linear checkbox

# sets the coordinates of the colormap axes. default is [0,1,0,1].
extent = [H.min(),H.max(),E.min(),E.max()]

# notes:
#   - signal has H, E as its 1st, 2nd axes. imshow sets the y-axis to the 1st one, so i
# transpose to put H on the x-axis. 
#   - aspect = 'auto' changes the colormap aspect ratio to fit the figure. 
# the default ('equal') fixes the aspect so that each pixel is equal size and the plot 
# aspect ratio is constrained. dont use the default.
#   - origin = 'lower' puts the origin of the plot at the lower corner. since arrays start
# at the upper left, the default puts the origin of the plot at the upper left corner. we 
# want it at the lower left corner
#   - norm object sets initial colorscale

# plot intensity
im_1 = ax[0].imshow(signal.T,cmap=cmap,aspect='auto',origin='lower',extent=extent,
                  norm=norm,interpolation='none')
# plot error
im_2 = ax[1].imshow(error.T,cmap=cmap,aspect='auto',origin='lower',extent=extent,
             norm=norm,interpolation='none')
# add colorbar
cbar = fig.colorbar(im_1,ax=[ax[0],ax[1]],extend='both',location='right')

# set axis labels and title etc
ax[0].set_xlabel('K [rlu]')
ax[1].set_xlabel('K [rlu]')
ax[0].set_ylabel('Energy [meV]')
ax[0].set_title('intensity')
ax[1].set_title('error')
fig.suptitle('Q=(6,K,0)',fontsize='x-large')

# --------------------------------------------------------------------------------------------------
# functions to update figure when textbox is changed/ checkbox is clicked
# they must be placed down here since they need im_1, im_2, etc. to be defined.

def update_maxbox(val):
    """
    callback passed to the vmax TextBox widget. the widget creates the float val and passes 
    to this func
    """
    # update the colormap vmax
    im_1.norm.vmax = val
    im_2.norm.vmax = val

    # redraw the figure
    fig.canvas.draw_idle()

def update_minbox(val):
    """
    callback passed to the vmax TextBox widget. the widget creates the float val and passes 
    to this func
    """
    # update the colormap vmin
    im_1.norm.vmin = val
    im_2.norm.vmin = val

    # redraw the figure 
    fig.canvas.draw_idle()

# this must be placed after the plots are created
def change_norm(val):
    """
    toggle between linear and symmetric-logarithm normalization of colormap ... SymLogNorm 
    sets scale to linear around 0 to avoid divergence and allow negative values
    """
    # get the status of check box
    log = logbox.get_status()[0]

    vmin = im_1.norm.vmin
    vmax = im_1.norm.vmax

    if log:
        linthresh = np.abs(vmax-vmin)/1e6 # linear region around 0
        norm = colors.SymLogNorm(linthresh=linthresh,linscale=1.0,vmin=vmin,vmax=vmax)
    else:
        norm = colors.Normalize(vmin,vmax)

    # reset the norms
    im_1.norm = norm
    im_2.norm = norm

    # redraw the figure to ensure it updates
    fig.canvas.draw_idle()

# --------------------------------------------------------------------------------------------------
# set what happends when clicked / text is changed etc

# set what is done when text is changed
#maxbox.on_text_change(update_maxbox) # changes as soon as text changes
maxbox.on_submit(update_maxbox) # press enter
#minbox.on_text_change(update_minbox) # changes as soon as text changes
minbox.on_submit(update_minbox) # press enter
logbox.on_clicked(change_norm)

# show the figure
plt.show()



