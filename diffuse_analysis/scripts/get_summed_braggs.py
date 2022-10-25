
from analysis_module.nds_data_set import rutile_dataset


file_names = ['reduced_data/050C_flashing_CC_initial.hdf5',
            'reduced_data/100C_flashing_CC_000mA.hdf5',
            'reduced_data/100C_flashing_CC_040mA.hdf5',
            'reduced_data/100C_flashing_CC_300mA.hdf5',
            'reduced_data/100C_flashing_CC_500mA.hdf5',
            'reduced_data/100C_flashing_Tot_040mA.hdf5',
            'reduced_data/100C_flashing_Tot_300mA.hdf5',
            'reduced_data/100C_flashing_Tot_500mA.hdf5',
            'reduced_data/400C_flashing_CC_000mA.hdf5',
            'reduced_data/400C_flashing_CC_500mA.hdf5',
            'reduced_data/400C_flashing_CC_950mA.hdf5',
            'reduced_data/400C_flashing_CC_initial.hdf5',
            'reduced_data/400C_flashing_Tot_000mA.hdf5',
            'reduced_data/400C_flashing_Tot_500mA.hdf5',
            'reduced_data/400C_Tot.hdf5']

labels = ['050C_flashing_CC_initial',
            '100C_flashing_CC_000mA',
            '100C_flashing_CC_040mA',
            '100C_flashing_CC_300mA',
            '100C_flashing_CC_500mA',
            '100C_flashing_Tot_040mA',
            '100C_flashing_Tot_300mA',
            '100C_flashing_Tot_500mA',
            '400C_flashing_CC_000mA',
            '400C_flashing_CC_500mA',
            '400C_flashing_CC_950mA',
            '400C_flashing_CC_initial',
            '400C_flashing_Tot_000mA',
            '400C_flashing_Tot_500mA',
            '400C_Tot']

raw_data = ['raw_data/DataNormalized_Symm_050C_flashing_CC_initial.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_CC_000mA.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_CC_040mA.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_CC_300mA.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_CC_500mA.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_Tot_040mA.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_Tot_300mA.nxs',
            'raw_data/DataNormalized_Symm_100C_flashing_Tot_500mA.nxs',
            'raw_data/DataNormalized_Symm_400C_flashing_CC_000mA.nxs',
            'raw_data/DataNormalized_Symm_400C_flashing_CC_500mA.nxs',
            'raw_data/DataNormalized_Symm_400C_flashing_CC_950mA.nxs',
            'raw_data/DataNormalized_Symm_400C_flashing_CC_initial.nxs',
            'raw_data/DataNormalized_Symm_400C_flashing_Tot_000mA.nxs',
            'raw_data/DataNormalized_Symm_400C_flashing_Tot_500mA.nxs',
            'raw_data/DataNormalized_Symm_400C_Tot.nxs']


n_dsets = len(labels)
for ii in range(n_dsets):

    file_name = file_names[ii]; label = labels[ii]; raw_file = raw_data[ii]

    print(file_name,'\n',label,'\n',raw_file,'\n')

    data_set = rutile_dataset(file_name,label)
#    data_set.get_from_nxs(raw_file)

    data_set.load_signal()
    data_set.get_bg()
    data_set.sum_good_braggs(bragg_list='reduced_braggs',
                         out_file=f'{label}_summed.hdf5',
                         flag_cut=0,
                         subtract_bg=True)





