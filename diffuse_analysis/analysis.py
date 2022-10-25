
from analysis_module.nds_data_set import rutile_dataset
from analysis_module.plot_module import plotting


file_name = 'reduced_data/rutile_293K_annealed.hdf5'
label = '293K_annealed'

data_set = rutile_dataset(file_name,label)

data_set.load_signal()
data_set.sum_good_braggs(out_file=f'{label}_summed.hdf5')

data_set.load_summed_signal(f'{label}_summed.hdf5')

plot = plotting(data_set)
plot.plot_summed_vol()

