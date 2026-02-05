
import numpy as np
import matplotlib.pyplot as plt


blue = '#377eb8'
orange = '#ff7f00'
green = '#4daf4a'


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

def get_window_avg(data,window_size=5):

    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Convolve the data with the kernel
    data = np.convolve(data, kernel, mode='same')

    return data

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4.5,4.5))

time, voltage, current, resistance = get_data('tio2_prist_600C.log')
time /= 60

# current = get_window_avg(current,window_size=10)
inds = np.flatnonzero(time < 2.35)
current[inds] = 0.0

# -----------------------------

ax.plot(time,voltage,lw=1.5,c=blue,zorder=0)
tax = ax.twinx()
tax.plot(time,current,lw=1.5,c=orange,zorder=10)
tax.axhline(0,lw=0.5,ls='--',c=(0.5,0.5,0.5))


tax_ins = tax.inset_axes([0.5,0.5,0.45,0.45],
                           xlim=(2.3,2.6),ylim=(-0.25,10),yticklabels=[],xticklabels=[])
tax.indicate_inset_zoom(tax_ins,edgecolor='black',alpha=1)

# current = get_window_avg(current,window_size=3)
tax_ins.plot(time,current,lw=1.5,c=orange,zorder=10)

# -----------------------------

for _ax in [ax,tax,tax_ins]:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize=12)
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.set_rasterization_zorder = 1000000000

# ax.annotate('(a)',xy=(0.025,0.92),xycoords='axes fraction',annotation_clip=False,fontsize=12)

# -----------------------------

ax.set_xlim(-0.1,10.5)
tax.set_xlim(-0.1,10.5)
ax.set_ylim(-30,910)
tax.set_ylim(-0.5,20)

ax.tick_params(axis='y',which='both',labelcolor=blue)

tax.tick_params(which='both',labelcolor=orange)
tax.set_ylabel('current density\n[A/cm$^2$]',color=orange,fontsize=16)

ax.set_ylabel('electric field\n[V/cm]',fontsize=16,labelpad=5,color=blue)
ax.set_xlabel('time [m]',fontsize=16,labelpad=5)

# -----------------------------

plt.savefig(f'tio2_flash_time_series.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()
