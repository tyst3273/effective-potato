################################
#
# Copy Facilities.xml to your 
# .mantid/instrument folder
#
################################


from m_save_MDE_to_hdf5 import save_MDE_to_hdf5

from mantid.simpleapi import *
import glob
import numpy as np

files = sorted(glob.glob('/SNS/ARCS/IPTS-26347/shared/jpark/NXSPE_Ebin0p5/S0*.nxspe'))
UBpars={'a':3.93,'b':3.93,'c':3.93,'alpha':90,'beta':90,'gamma':90,'u':'1,0,0','v':'0,1,0'}

fn = '/SNS/ARCS/IPTS-26347/shared/jpark/120meV300Hz010K_run_list.txt'
d=np.loadtxt(fn,skiprows=1,dtype=str)
runs=d[:,0]
gon=d[:,11].astype(float)-82.388
gon_dict=dict()
for r,g in zip(runs, gon):
    gon_dict[r]=g

wg=Load(','.join(files))
for wsn in wg.getNames():
    ang=gon_dict[wsn.split('S0')[1]]
    SetGoniometer(Workspace=wsn, Axis0 = f'{ang},0,1,0,1')
axis_deltaE=wg[0].readX(0)
emin=axis_deltaE[0]
emax=axis_deltaE[-1]
ConvertFromDistribution(wg)
wge=ConvertToEventWorkspace(wg)
wge=CropWorkspaceForMDNorm(wge,XMin=emin,XMax=emax)
SetUB(wge,**UBpars)
AddSampleLog(Workspace='wge', LogName='gd_prtn_chrg', LogText='1.0', LogType='Number', NumberType='Double')
mdparts=ConvertToMD(InputWorkspace=wge, QDimensions='Q3D', dEAnalysisMode='Direct', Q3DFrames='Q_sample')
mde=MergeMD(mdparts)
#SaveMD(InputWorkspace="mde",Filename="/SNS/ARCS/IPTS-26347/shared/jpark/MDEventFile.nxs")

UBpars={'a':3.93,'b':3.93,'c':3.93,'alpha':90,'beta':90,'gamma':90,'u':'1,0,-0.1','v':'0,1,-0.1'}
SetUB(mde,**UBpars)

# test MDNorm - elastic slice
MDNorm(InputWorkspace=mde,
       QDimension0='1,0,0',
       QDimension1='0,1,0',
       QDimension2='0,0,1',
       Dimension0Name='QDimension0',
       Dimension0Binning='0.05',
       Dimension1Name='QDimension1',
       Dimension1Binning='0.05',
       Dimension2Name='QDimension2',
       Dimension2Binning='0.05',
       Dimension3Name='DeltaE',
       Dimension3Binning='0.5',
       SymmetryOperations='x,y,z;-x,-y,z;x,y,-z;-x,-y,-z;y,x,z;y,x,-z',
       OutputWorkspace='o',
       OutputDataWorkspace='d',
       OutputNormalizationWorkspace='n')

SaveMD(InputWorkspace="o",Filename="/SNS/ARCS/IPTS-26347/shared/jpark/MDHistoFile.nxs")

MD_workspace = mtd['o']
hdf5_file_name = 'jpark.hdf5'
save_MDE_to_hdf5(MD_workspace,hdf5_file_name)





