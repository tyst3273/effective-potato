
from analysis_module.nds_data_set import rutile_dataset


file_name = 'reduced_data/rutile_400C_500mA.hdf5'
label = '400C_500mA'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.sum_good_braggs(bragg_list='reduced_braggs',out_file=f'{label}_summed.hdf5',
                    flag_cut=0,subtract_bg=True)

file_name = 'reduced_data/rutile_400C_950mA.hdf5'
label = '400C_950mA'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.sum_good_braggs(bragg_list='reduced_braggs',out_file=f'{label}_summed.hdf5',
                    flag_cut=0,subtract_bg=True)

file_name = 'reduced_data/rutile_100C_000mA.hdf5'
label = '100C_000mA'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.sum_good_braggs(bragg_list='reduced_braggs',out_file=f'{label}_summed.hdf5',
                    flag_cut=0,subtract_bg=True)

file_name = 'reduced_data/rutile_050C_initial.hdf5'
label = '050C_initial'
data_set = rutile_dataset(file_name,label)
data_set.load_signal()
data_set.sum_good_braggs(bragg_list='reduced_braggs',out_file=f'{label}_summed.hdf5',
                    flag_cut=0,subtract_bg=True)


