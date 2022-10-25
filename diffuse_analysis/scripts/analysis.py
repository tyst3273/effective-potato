
from analysis_module.nds_data_set import rutile_dataset
from analysis_module.plot_module import plotting


file_name = 'reduced_data/rutile_293K_quenched.hdf5'
label = '293K_quenched'
data_set = rutile_dataset(file_name,label)

#raw_file = '/media/ty/hitachi/archive/materials/rutile/raw_data/normData_293K_quenched_2nd.nxs'
#data_set.get_from_nxs(raw_file)

data_set.load_signal()
#data_set.get_bg()

#data_set.print_bragg_list()

#plot = plotting(data_set)
#plot.plot_zones(out_file=f'all_zones_H_{label}.pdf',
#        fixed_axis='H',bragg_file=None)
#plot.plot_zones(out_file=f'all_zones_K_{label}.pdf',
#        fixed_axis='K',bragg_file=None)
#plot.plot_zones(out_file=f'all_zones_L_{label}.pdf',
#        fixed_axis='L',bragg_file=None)

data_set.sum_good_braggs(bragg_list='reduced_braggs',out_file=f'{label}_summed.hdf5',flag_cut=0,subtract_bg=True)
data_set.load_summed_signal(f'{label}_summed.hdf5')

plot = plotting(data_set)
plot.plot_summed_vol()

