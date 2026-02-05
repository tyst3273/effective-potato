
import numpy as np
import matplotlib.pyplot as plt
import h5py


blue = '#377eb8'
orange = '#ff7f00'
green = '#4daf4a'

J2meV = 6.242e+18 * 1000

v0 = 2000
j0 = 2000

r0 = 20
T0 = 2000
C2K = 300

kb = 1.38e-23 # J / C
q = 1.6e-19 # coulombs

E0 = T0 * kb * J2meV # meV
print('E0 = ',E0,'meV')
Q0 = E0/v0/1000 # electron charge
print('Q0 = ',Q0,'e')


# --------------------------------------------------------------------------------------------------

def get_j_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    v = np.tile(v.reshape(v.size,1),reps=(1,3))
    v = v.flatten()
    x = x.flatten()
    n = n.flatten()
    j = v * n / x

    inds = np.flatnonzero( ~np.isnan(x) )
    j = j[inds]
    v = v[inds]
    x = x[inds]
    n = n[inds]

    inds = np.argsort(j)
    j = j[inds]
    v = v[inds]
    x = x[inds]
    n = n[inds]

    return n, x, v, j, y, z

# --------------------------------------------------------------------------------------------------

def get_calc_data(filename):

    with h5py.File(filename,'r') as db:
        time = db['time'][...]
        v = db['v'][...]
        j = db['j'][...]

    return time, v, j

# --------------------------------------------------------------------------------------------------

def get_data(filename):

    # TIME[S]       V-OUT[V]      I-OUT[A]      V-SET[V]      I-SET[A]      

    time, voltage, current, _, _ = np.loadtxt(filename,unpack=True,skiprows=1)

    ### intensive ###
    l = 0.35 # cm
    w = 0.28 # cm
    t = 0.05 # cm
    area = w*t # cm**2
    current /= area # amps / cm^2
    voltage /= l # v / cm
    resistivity = voltage/current # Ohm * cm
    ### intensive ###

    return time, voltage, current, resistivity

# --------------------------------------------------------------------------------------------------

def get_window_avg(data,window_size=50):

    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Convolve the data with the kernel
    data = np.convolve(data, kernel, mode='same')

    return data

# --------------------------------------------------------------------------------------------------

def get_data_prist(filename):

    # TIME[S]       I-SET[A]      V-NVM[V]      V-VMT[A]      TEMP[C]   compliance

    time, current, v4, vT, temp, _ = np.loadtxt(filename,unpack=True,skiprows=1)

    ### intensive ###
    l2 = 0.83 # cm
    l4 = 0.2 # cm
    w = 0.05 # cm
    t = 0.173 # cm
    area = w*t # cm**2
    current /= area # amps / cm^2
    v4 /= l4 # v / cm
    res = v4/current # Ohm * cm
    ### intensive ###

    return time, res, current, temp

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'wspace':0.2,'hspace':0.3})
tax = [ax[0,0].twinx(),ax[0,1].twinx()]

# --------------------------------------------------------------------------------------------------

time, voltage, current, resistance = get_data('tio2_prist_600C.log')
time /= 60

# current = get_window_avg(current,window_size=10)
inds = np.flatnonzero(time < 2.35)
current[inds] = 0.0

ax[0,0].plot(time,voltage,lw=1.5,c=blue,zorder=0)
tax[0].plot(time,current,lw=1.5,c=orange,zorder=10)
tax[0].axhline(0,lw=0.5,ls='--',c=(0.5,0.5,0.5))

# ax_ins = tax[0].inset_axes([0.5,0.5,0.45,0.45],
#                            xlim=(2.3,2.6),ylim=(-0.25,10),yticklabels=[],xticklabels=[])
# tax[0].indicate_inset_zoom(ax_ins,edgecolor='black',alpha=1)

# ax_ins.plot(time,current,lw=1.5,c=orange,zorder=10)

# -----------------------------

time, v, j = get_calc_data('time_series_y_0.050_z_0.500.h5')

time *= 9.8

v *= v0 
j *= j0 

ind = 140000
ax[0,1].plot(time[:ind],v[:ind],lw=2,c=blue,zorder=0)
ax[0,1].plot(time[ind:],v[ind:],lw=1,c=blue,zorder=0,ls=(0,(2,1,1,1)))
tax[1].plot(time,j,lw=2,c=orange,zorder=10)
tax[1].axhline(0,lw=0.5,ls='--',c=(0.5,0.5,0.5))
tax[1].axhline(0,lw=0.5,ls='--',c=(0.5,0.5,0.5))

# --------------------------------------------------------------------------------------------------

_time, _res, _current, _temp = get_data_prist('tio2_prist_cond.log')

cut = 11500 
lo_cut = 7500 #3750
_res = get_window_avg(_res)[lo_cut:cut]
_temp = get_window_avg(_temp)[lo_cut:cut]
ax[1,0].plot(_temp,_res,lw=1.5,c='k',zorder=-10,ls=(0,(4,1,2,1)))

# --------------------

curr, res = np.loadtxt('tio2_curr_vs_res.txt',unpack=True)
coeffs = [441.00745325,  77.24366005]
T = coeffs[0]+curr*coeffs[1]
ax[1,0].plot(T[:-1],res[:-1],marker='o',ms=3,c=green,lw=1,ls='--')

ax[1,0].annotate('equilibrium',xy=(0.5,0.55),xycoords='axes fraction',fontsize=12)
ax[1,0].annotate('in-situ',xy=(0.4,0.2),xycoords='axes fraction',fontsize=12,c=green)

# --------------------------------------------------------------------------------------------------

y = 0.05
z_list = [0.5,1.0]
colors = [orange,green]

x_min = 1e9
x_max = 0

for ii, zz in enumerate(z_list):

    ax[1,1].plot(-1,0,c=colors[ii],lw=2,label=f'z={zz:.3f}')

    n, x, v, j, y, z = get_j_data(f'results_v_y_{y:.3f}_z_{zz:.3f}.h5')

    if np.nanmin(x) < x_min:
        x_min = np.nanmin(x)
    if np.nanmax(x) > x_max:
        x_max = np.nanmax(x)
    
    r = x / n
    ax[1,1].plot(x*T0+C2K,r * r0, c=colors[ii],lw=2,marker='o',ms=0)
    ax[1,1].plot(x*T0+C2K,r * r0, c=colors[ii],lw=2,marker='o',ms=0)

x = np.linspace(x_min,x_max,10001)
r_eq = x / np.exp(-1/x) 
ax[1,1].plot(x*T0+C2K,r_eq * r0 ,c='k',ls=(0,(2,1,1,1)),zorder=100)

ax[1,1].annotate('equilibrium\n  ($\zeta$=0 e)',xy=(0.4,0.5),xycoords='axes fraction',fontsize=12)

zeta = 1 * Q0
ax[1,1].annotate(f'$\zeta$={zeta:.1e} e',xy=(0.35,0.4),xytext=(0.05,0.2),xycoords='axes fraction',fontsize=12,
               color=green,arrowprops=dict(color=green,lw=1,arrowstyle='->'))
zeta = 0.5 * Q0
ax[1,1].annotate(f'$\zeta$={zeta:.1e} e',xy=(0.7,0.325),xytext=(0.45,0.05),xycoords='axes fraction',fontsize=12,
               color=orange,arrowprops=dict(color=orange,lw=1,arrowstyle='->'))

# -----------------------------

for ii in range(2):
    for jj in range(2):
        _ax = ax[ii,jj]
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize=12)
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterization_zorder = 1000000000

for _ax in [*tax]: #,ax_ins]:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize=12)
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.set_rasterization_zorder = 1000000000

# ax.annotate('(a)',xy=(0.025,0.92),xycoords='axes fraction',annotation_clip=False,fontsize=12)

# -----------------------------

ax[0,0].axis([-0.1,10.5,-30,1000])
tax[0].axis([-0.1,10.5,-0.5,20])

ax[0,1].axis([-0.1,10.5,-30,1000])
tax[1].axis([-0.1,10.5,-0.5,20])

ax[0,0].tick_params(axis='y',which='both',labelcolor=blue)
ax[0,1].tick_params(axis='y',which='both',labelcolor=blue)

tax[0].tick_params(which='both',labelcolor=orange)
tax[1].tick_params(which='both',labelcolor=orange)

tax[1].set_ylabel('current density [A/cm$^2$]',color=orange,fontsize=16,y=0.425)
ax[0,0].set_ylabel('electric field [V/cm]',fontsize=16,labelpad=5,color=blue)
ax[0,0].set_xlabel('time [m]',fontsize=16,labelpad=0)
ax[0,1].set_xlabel('time [m]',fontsize=16,labelpad=0)

tax[0].set_yticklabels([])
ax[0,1].set_yticklabels([])

# -----------------------------

ax[1,0].set_xlabel('T [C]',fontsize=16,labelpad=0)
ax[1,0].set_ylabel(r'resistivity [$\Omega$-cm]',color='k',fontsize=16)

ax[1,0].set_ylim(10,1e5)
ax[1,0].set_xlim(500,950)
ax[1,0].set_yscale('log')

ax[1,1].set_ylabel(r'resistivity [$\Omega$-cm]',fontsize=16,position='right')
ax[1,1].set_xlabel('T [C]',fontsize=16,labelpad=0)

# ax[1,1].set_ylim(1,100)
# ax[1,1].set_xlim(0.1,0.55)

ax[1,1].set_xlim(500,950)
ax[1,1].set_ylim(10,1e5)

ax[1,1].set_yscale('log')

ax[1,1].yaxis.tick_right()
ax[1,1].yaxis.set_label_position('right')

# -----------------------------

ax[0,0].annotate('(a)',xy=(0.025,0.925),xycoords='axes fraction',c='k',fontsize=12)
ax[0,1].annotate('(b)',xy=(0.025,0.925),xycoords='axes fraction',c='k',fontsize=12)
ax[1,0].annotate('(c)',xy=(0.025,0.925),xycoords='axes fraction',c='k',fontsize=12)
ax[1,1].annotate('(d)',xy=(0.025,0.925),xycoords='axes fraction',c='k',fontsize=12)

plt.savefig(f'tio2_exp_vs_model.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()
