
from analysis_module.nds_data_set import rutile_dataset
from analysis_module.plot_module import plotting


file_name = 'reduced_data/rutile_400C_500mA.hdf5'
label = '400C_500mA'
data_set = rutile_dataset(file_name,label)

#raw_file = 'raw_data/DataNormalized_Symm_050C_flashing_CC_initial.nxs'
#data_set.get_from_nxs(raw_file)

data_set.load_signal()
#data_set.get_bg()
#data_set.print_bragg_list()

#plot = plotting(data_set)
#plot.plot_zones(out_file=f'reduced_zone_H_{label}.pdf',
#        fixed_axis='H',bragg_file='braggs/reduced_braggs')
#plot.plot_zones(out_file=f'reduced_zone_K_{label}.pdf',
#        fixed_axis='K',bragg_file='braggs/reduced_braggs')
#plot.plot_zones(out_file=f'reduced_zone_L_{label}.pdf',
#        fixed_axis='L',bragg_file='braggs/reduced_braggs')

#data_set.sum_good_braggs(bragg_list='reduced_braggs',out_file=f'{label}_summed.hdf5',flag_cut=0,subtract_bg=True)

data_set.load_summed_signal(f'{label}_summed.hdf5')

plot = plotting(data_set)
plot.plot_summed_vol()

