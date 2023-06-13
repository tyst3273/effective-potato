
from diffit.m_code_utils import c_timer
from psf.m_PSF import c_PSF

# --------------------------------------------------------------------------------------------------

def run_PSF(crystal,input_file=None,**kwargs):

    """
    setup and run PSF
    """

    timer = c_timer('PSF')

    # default kwargs 
    default_kwargs = {'output_prefix':None,'calc_sqw':False,'box_vectors':None,'unwrap_trajectory':False}

    # overwrite defaults
    default_kwargs.update(kwargs)
    
    PSF = c_PSF(input_file)

    # setup PSF calculation
    PSF.setup_calculation(pos=crystal.sc_positions_cart,
                          types=crystal.sc_type_nums,
                          md_num_atoms=crystal.num_sc_atoms,
                          lattice_vectors=crystal.basis_vectors,
                          md_num_steps=1,
                          md_time_step=1,
                          trajectory_format='external',
                          **default_kwargs)

    PSF.run()

    calculated_intensity = PSF.comm.strufacs.sq_elastic

    timer.stop()

    return calculated_intensity

# --------------------------------------------------------------------------------------------------

