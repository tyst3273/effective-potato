import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from analysis_module.analysis_utils import _timer

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"


class plotting:

    def __init__(self,data_set):

        """
        store data for this dataset
        """

        self.data_set = data_set

        self.cmap = 'binary'
        self.interp = 'none'
        self.vmin = 0
        self.vmax = 1
        self.eps = 0.02
        self.dH = 0.011
        self.dK = 0.041
        self.dL = 0.011


    def plot_bragg_planes(self,out_file=None,fixed_axis='K',bragg_file=None):

        """
        plot all bragg planes.
        """

        timer = _timer('plot_bragg_planes')

        data_set = self.data_set
        data_set.get_bragg_peaks(bragg_file)

        if out_file is None:
            out_file = f'bragg_planes_{fixed_axis}_{data_set.label}.pdf'
        pdf_doc = PdfPages(out_file)

        if fixed_axis == 'H':
            braggs = data_set.H_bragg
            n_planes = braggs.size
        elif fixed_axis == 'K':
            braggs = data_set.K_bragg
            n_planes = braggs.size
        elif fixed_axis == 'L':
            braggs = data_set.L_bragg
            n_planes = braggs.size
        else:
            msg = f'\nERROR!\n axis \'{fixed_axis}\' unknown'
            exit(msg)

        for ii in range(n_planes):

            if fixed_axis == 'H':   

                Q = data_set.H_bragg[ii]
                title = r'$\bm{Q}$='+f'({Q:3.2f}, K, L) r.l.u.'
                x_label = 'K (r.l.u.)'; y_label = 'L (r.l.u.)'
                inds = np.arange(np.argwhere(data_set.H >= Q-self.dH).flatten()[0],
                    np.argwhere(data_set.H <= Q+self.dH).flatten()[-1])
                _sig = data_set.signal[inds,:,:].mean(axis=0).T
                _bg = data_set.bg[inds,:,:].mean(axis=0).T
                extent = [data_set.K.min(),data_set.K.max(),data_set.L.min(),data_set.L.max()]
                
            if fixed_axis == 'K':

                Q = data_set.K_bragg[ii]
                title = r'$\bm{Q}$='+f'(H, {Q:3.2f}, L) r.l.u.'
                x_label = 'H (r.l.u.)'; y_label = 'L (r.l.u.)'
                inds = np.arange(np.argwhere(data_set.K >= Q-self.dK).flatten()[0],
                    np.argwhere(data_set.K <= Q+self.dK).flatten()[-1])
                _sig = data_set.signal[:,inds,:].mean(axis=1).T
                _bg = data_set.bg[:,inds,:].mean(axis=1).T
                extent = [data_set.H.min(),data_set.H.max(),data_set.L.min(),data_set.L.max()]

            if fixed_axis == 'L':

                Q = data_set.L_bragg[ii]
                title = r'$\bm{Q}$='+f'(H, K, {Q:3.2f}) r.l.u.'
                x_label = 'K (r.l.u.)'; y_label = 'L (r.l.u.)'
                inds = np.arange(np.argwhere(data_set.L >= Q-self.dL).flatten()[0],
                    np.argwhere(data_set.L <= Q+self.dL).flatten()[-1])
                _sig = data_set.signal[:,:,inds].mean(axis=2).T
                _bg = data_set.bg[:,:,inds].mean(axis=2).T
                extent = [data_set.H.min(),data_set.H.max(),data_set.L.min(),data_set.L.max()]

            print(f'Q={Q:3.2f}')

            fig, ax = plt.subplots(1,3,figsize=(11.5,3),gridspec_kw={'wspace':0.15})
            ax[0].imshow(_sig,aspect='auto',origin='lower',vmin=self.vmin,vmax=self.vmax,
                    cmap=self.cmap,interpolation=self.interp,extent=extent)
            ax[1].imshow(_bg,aspect='auto',origin='lower',vmin=self.vmin,vmax=self.vmax,
                    cmap=self.cmap,interpolation=self.interp,extent=extent)
            im = ax[2].imshow(_sig-_bg,aspect='auto',origin='lower',vmin=self.vmin,
                    vmax=self.vmax,cmap=self.cmap,interpolation=self.interp,extent=extent)
            fig.colorbar(im,ax=ax,extend='both',pad=0.02)

            for ii in range(3):
                for axis in ['top','bottom','left','right']:
                    ax[ii].spines[axis].set_linewidth(1.5)

                ax[ii].minorticks_on()
                ax[ii].tick_params(which='both',width=1,labelsize='large')
                ax[ii].tick_params(which='major',length=5)
                ax[ii].tick_params(which='minor',length=2)
                x_major_locator = MultipleLocator(2)
                y_major_locator = MultipleLocator(2)
                x_minor_locator = MultipleLocator(1)
                y_minor_locator = MultipleLocator(1)
                x_major_formatter = FormatStrFormatter('%2.1f')
                y_major_formatter = FormatStrFormatter('%2.1f')
                ax[ii].set_xlabel(x_label,fontsize='x-large')

            fig.suptitle(title,fontsize='x-large',y=1.05)
            ax[0].set_title('raw',fontsize='x-large')
            ax[1].set_title('BG',fontsize='x-large')
            ax[2].set_title('raw-BG',fontsize='x-large')
            ax[0].set_ylabel(y_label,labelpad=-1,fontsize='x-large')
            ax[1].set_yticklabels([])
            ax[2].set_yticklabels([])

            pdf_doc.savefig(fig,dpi=150,bbox_inches='tight')
            plt.close()

        pdf_doc.close()
        timer.stop()


    def _set_grids(self,ax,major=1,minor=0.5):
        
        """
        set grids on an mpl axis object
        """
        
        x_major_locator = MultipleLocator(major)
        y_major_locator = MultipleLocator(major)
        x_minor_locator = MultipleLocator(minor)
        y_minor_locator = MultipleLocator(minor)
        x_major_formatter = FormatStrFormatter('%2.1f')
        y_major_formatter = FormatStrFormatter('%2.1f')

        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_major_formatter(x_major_formatter)
        ax.xaxis.set_minor_locator(x_minor_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_major_formatter(y_major_formatter)
        ax.xaxis.set_minor_locator(y_minor_locator)
        ax.xaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
        ax.xaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
        ax.yaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
        ax.yaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
        ax.xaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
        ax.xaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
        ax.yaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
        ax.yaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
        ax.xaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
        ax.xaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
        ax.yaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
        ax.yaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)


    def plot_zones(self,out_file=None,fixed_axis='K',bragg_file=None):

        """
        plot all bragg planes; plot each zone seprately
        """

        timer = _timer('plot_all_zones')
        data_set = self.data_set
        data_set.get_bragg_peaks(bragg_file)
        n_peaks = data_set.bragg_peaks.shape[0]

        if out_file is None:
            out_file = f'zones_{fixed_axis}_{data_set.label}.pdf'
        pdf_doc = PdfPages(out_file)

        for ii in range(n_peaks):

            print('ii = ',ii)
            H = data_set.bragg_peaks[ii,0]; K = data_set.bragg_peaks[ii,1]; L = data_set.bragg_peaks[ii,2]
            title = r'$\bm{Q}$='+f'({H:3.2f}, {K:3.2f}, {L:3.2f}) r.l.u.'

            if fixed_axis == 'H':

                x_label = 'K (r.l.u.)'; y_label = 'L (r.l.u.)'
                H_inds = np.arange(np.argwhere(data_set.H >= H-self.dH).flatten()[0],
                    np.argwhere(data_set.H <= H+self.dH).flatten()[-1])
                K_inds = np.arange(np.argwhere(data_set.K >= K-(0.5+self.dK)).flatten()[0],
                    np.argwhere(data_set.K <= K+(0.5+self.dK)).flatten()[-1])
                L_inds = np.arange(np.argwhere(data_set.L >= L-(0.5+self.dL)).flatten()[0],
                    np.argwhere(data_set.L <= L+(0.5+self.dL)).flatten()[-1])
                _sig = data_set.signal[H_inds,:,:]
                _sig = _sig[:,K_inds,:]
                _sig = _sig[:,:,L_inds]
                _sig = _sig.mean(axis=0).T
                _bg = data_set.bg[H_inds,:,:]
                _bg = _bg[:,K_inds,:]
                _bg = _bg[:,:,L_inds]
                _bg = _bg.mean(axis=0).T

                extent = [data_set.K[K_inds].min(),data_set.K[K_inds].max(),
                            data_set.L[L_inds].min(),data_set.L[L_inds].max()]
                lims = [K-0.5,K+0.5,L-0.5,L+0.5]

            if fixed_axis == 'K':

                x_label = 'H (r.l.u.)'; y_label = 'L (r.l.u.)'
                H_inds = np.arange(np.argwhere(data_set.H >= H-(0.5+self.dH)).flatten()[0],
                    np.argwhere(data_set.H <= H+(0.5+self.dH)).flatten()[-1])
                K_inds = np.arange(np.argwhere(data_set.K >= K-self.dK).flatten()[0],
                    np.argwhere(data_set.K <= K+self.dK).flatten()[-1])
                L_inds = np.arange(np.argwhere(data_set.L >= L-(0.5+self.dL)).flatten()[0],
                    np.argwhere(data_set.L <= L+(0.5+self.dL)).flatten()[-1])
                _sig = data_set.signal[H_inds,:,:]
                _sig = _sig[:,K_inds,:]
                _sig = _sig[:,:,L_inds]
                _sig = _sig.mean(axis=1).T
                _bg = data_set.bg[H_inds,:,:]
                _bg = _bg[:,K_inds,:]
                _bg = _bg[:,:,L_inds]
                _bg = _bg.mean(axis=1).T

                extent = [data_set.H[H_inds].min(),data_set.H[H_inds].max(),
                            data_set.L[L_inds].min(),data_set.L[L_inds].max()]
                lims = [H-0.5,H+0.5,L-0.5,L+0.5]

            if fixed_axis == 'L':
                
                x_label = 'K (r.l.u.)'; y_label = 'K (r.l.u.)'
                H_inds = np.arange(np.argwhere(data_set.H >= H-(0.5+self.dH)).flatten()[0],
                    np.argwhere(data_set.H <= H+(0.5+self.dH)).flatten()[-1])
                K_inds = np.arange(np.argwhere(data_set.K >= K-(0.5+self.dK)).flatten()[0],
                    np.argwhere(data_set.K <= K+(0.5+self.dK)).flatten()[-1])
                L_inds = np.arange(np.argwhere(data_set.L >= L-self.dL).flatten()[0],
                    np.argwhere(data_set.L <= L+self.dL).flatten()[-1])
                _sig = data_set.signal[H_inds,:,:]
                _sig = _sig[:,K_inds,:]
                _sig = _sig[:,:,L_inds]
                _sig = _sig.mean(axis=2).T
                _bg = data_set.bg[H_inds,:,:]
                _bg = _bg[:,K_inds,:]
                _bg = _bg[:,:,L_inds]
                _bg = _bg.mean(axis=2).T

                extent = [data_set.H[H_inds].min(),data_set.H[H_inds].max(),
                            data_set.K[K_inds].min(),data_set.K[K_inds].max()]
                lims = [H-0.5,H+0.5,K-0.5,K+0.5]

            fig, ax = plt.subplots(figsize=(6,5),gridspec_kw={'wspace':0.15})
            im = ax.imshow(_sig-_bg,aspect='auto',origin='lower',vmin=self.vmin,
                    vmax=self.vmax,cmap=self.cmap,interpolation=self.interp,extent=extent)
            fig.colorbar(im,ax=ax,extend='both',pad=0.02)

            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)

            ax.axis(lims)
            ax.minorticks_on()
            ax.tick_params(which='both',width=1,labelsize='large')
            ax.tick_params(which='major',length=5)
            ax.tick_params(which='minor',length=2)
            self._set_grids(ax,major=0.5,minor=0.1)

            ax.set_title(title,fontsize='x-large',pad=20.0)
            ax.set_xlabel(x_label,fontsize='x-large')
            ax.set_ylabel(y_label,labelpad=5,fontsize='x-large')

            pdf_doc.savefig(fig,dpi=150,bbox_inches='tight')
            plt.close()

        pdf_doc.close()
        timer.stop()


    def plot_summed_vol(self):
        
        """
        plot volumetric data using mayavi
        """

        timer = _timer('plot_vol')

        h = self.data_set.h; k = self.data_set.k; l = self.data_set.l
        summed_signal = self.data_set.summed_signal

        if not hasattr(self.data_set,'summed_signal'):
            msg = ' ERROR!\n the data set doesnt seem to have \'summed_signal\' attribute.\n' \
                  ' have you loaded it yet (using load_summed_signal...)\n'
            exit(msg)

        try:
            from mayavi import mlab
        except Exception as _ex:
            print(_ex)
            msg = ' ERROR!\n mayavi couldnt be imported. make sure it is \n' \
                  ' installed and the correct environment is activated\n'
            exit(msg)

        fig = mlab.figure(1, bgcolor=(1,1,1), fgcolor=(0,0,0),size=(500, 500))
        mlab.clf()

        extent = [h.min(),h.max(),k.min(),k.max(),l.min(),l.max()]
        h, k, l = np.meshgrid(h,k,l,indexing='ij')

        """
        contours = []
        for ii in np.linspace(0.01,0.05,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,0.5,1),
                transparent=True,opacity=0.001,figure=fig)
        contours = []
        for ii in np.linspace(0.04,0.1,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,0.5,1),
                transparent=True,opacity=0.01,figure=fig)
        contours = []
        for ii in np.linspace(0.075,0.5,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,1,0),
                transparent=True,opacity=1.0,figure=fig)
        """

        contours = []
        for ii in np.linspace(0.285,0.325,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,0.5,1),
                transparent=True,opacity=0.0025,figure=fig)
        contours = []
        for ii in np.linspace(0.325,0.4,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,0.5,1),
                transparent=True,opacity=0.01,figure=fig)
        contours = []
        for ii in np.linspace(0.375,1.0,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,1,0),
                transparent=True,opacity=1.0,figure=fig)

        mlab.outline(color=(0,0,0),line_width=2.5,extent=extent)
        mlab.axes(color=(0,0,0),line_width=2.5,nb_labels=5,
            xlabel='H (r.l.u.)',
            ylabel='K (r.l.u.)',
            zlabel='L (r.l.u.)')

        fig.scene.parallel_projection = True
        mlab.orientation_axes()

        mlab.show()
        timer.stop()


    def _plot_summed_vol(self):

        """
        plot volumetric data using mayavi
        """

        timer = _timer('plot_vol')

        h = self.data_set.h; k = self.data_set.k; l = self.data_set.l
        summed_signal = self.data_set.summed_signal

        if not hasattr(self.data_set,'summed_signal'):
            msg = ' ERROR!\n the data set doesnt seem to have \'summed_signal\' attribute.\n' \
                  ' have you loaded it yet (using load_summed_signal...)\n'
            exit(msg)

        try:
            from mayavi import mlab
        except:
            msg = ' ERROR!\n mayavi couldnt be imported. make sure it is \n' \
                  ' installed and the correct environment is activated\n'
            exit(msg)

        fig = mlab.figure(1, bgcolor=(1,1,1), fgcolor=(0,0,0),size=(500, 500))
        mlab.clf()

        extent = [h.min(),h.max(),k.min(),k.max(),l.min(),l.max()]
        h, k, l = np.meshgrid(h,k,l,indexing='ij')

        contours = []
        for ii in np.linspace(0.2,0.4,200):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,0.5,1),
                transparent=True,opacity=0.01,figure=fig)
        contours = []
        for ii in np.linspace(0.35,2.0,100):
            contours.append(ii)
        mlab.contour3d(h,k,l,summed_signal,contours=contours,color=(1,1,0),
                transparent=True,opacity=1.0,figure=fig)


        mlab.outline(color=(0,0,0),line_width=2.5,extent=extent)
        mlab.axes(color=(0,0,0),line_width=2.5,nb_labels=5,
            xlabel='H (r.l.u.)',
            ylabel='K (r.l.u.)',
            zlabel='L (r.l.u.)')

        fig.scene.parallel_projection = True
        mlab.orientation_axes()

        mlab.show()
        timer.stop()

    
    





        
