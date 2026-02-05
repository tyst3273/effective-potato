
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

def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    return n, x, v, y, z

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

fig, ax = plt.subplots(figsize=(4.5,4.5),gridspec_kw={'wspace':0.2,'hspace':0.2})

# -------------------------------

n, x, v, y, z = get_data(f'results_v_y_0.050_z_0.500.h5')

j = n / x
j[:,0] *= v
j[:,1] *= v
j[:,2] *= v

v *= v0
j *= j0

ax.plot(v,j[:,0],c=green,lw=0,marker='o',ms=1.5)
ax.plot(v,j[:,1],c=green,lw=1,marker='o',ms=0)
ax.plot(v,j[:,2],c=green,lw=0,marker='o',ms=1.5)

# --------------------------------------------------------------------------------------------------

_ax = ax
for axis in ['top','bottom','left','right']:
    _ax.spines[axis].set_linewidth(1.5)
_ax.minorticks_on()
_ax.tick_params(which='both',width=1,labelsize=12)
_ax.tick_params(which='major',length=5)
_ax.tick_params(which='minor',length=2)
_ax.set_rasterization_zorder = 1000000000

# -----------------------------

ax.set_ylabel('current density [A/cm$^2$]',fontsize=16)
ax.set_xlabel('electric field [V/cm]',fontsize=16)

ax.set_xlim(-30,1000)
ax.set_ylim(1e-8,1000)

ax.set_yscale('log')

# -----------------------------

T_bath = y * T0 + C2K

# E0 = T1 * kb
# print('E0 =',E0*J2meV,'meV')

# z0 = E0/v0
# zeta = z * z0
# print(zeta / q)

ax.annotate(f'T$_0$={T_bath:.1f} [C]',xy=(0.025,0.925),xycoords='axes fraction',c='k',fontsize=12)

zeta = 0.5 * Q0
ax.annotate(f'$\zeta$={zeta:.1e} e',xy=(0.025,0.875),xycoords='axes fraction',c='k',fontsize=12)

plt.savefig(f'iv.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()
