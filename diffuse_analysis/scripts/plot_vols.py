
from analysis_module.nds_data_set import rutile_dataset
from analysis_module.plot_module import plotting


file_name = 'reduced_data/rutile_400C_500mA.hdf5'
label = '400C_500mA'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.load_summed_signal(f'{label}_summed.hdf5')
plot = plotting(data_set)
plot.plot_summed_vol()

file_name = 'reduced_data/rutile_400C_950mA.hdf5'
label = '400C_950mA'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.load_summed_signal(f'{label}_summed.hdf5')
plot = plotting(data_set)
plot.plot_summed_vol()

file_name = 'reduced_data/rutile_100C_000mA.hdf5'
label = '100C_000mA'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.load_summed_signal(f'{label}_summed.hdf5')
plot = plotting(data_set)
plot.plot_summed_vol()

file_name = 'reduced_data/rutile_050C_initial.hdf5'
label = '050C_initial'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.load_summed_signal(f'{label}_summed.hdf5')
plot = plotting(data_set)
plot.plot_summed_vol()

