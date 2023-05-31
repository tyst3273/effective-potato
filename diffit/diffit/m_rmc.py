
import numpy as np
from diffit.m_code_utils import crash, c_timer


class c_rmc:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,beta=0.5,exit_tol=1e-3):

        """
        class to calculate agreement with exp data and accept or reject move according
        to Boltzmann weight
        """
       
        self.beta = beta
        self.exit_tol = exit_tol

        # initialize these
        self.converged = False
        self.error_squared = 0
        self.delta_error_squared = 0
        
    # ----------------------------------------------------------------------------------------------

    def check_move(self,exp_data,calc_data):

        """
        calculate error squared between exp data and calculated data and get boltzmann weight.
        accept or reject move according to whether or not random number in interval [0,1] is 
        less than:

            weight = exp( -error_sq * beta / 2) 
        
        """

        new_error_sq = np.sum((exp_data-calc_data)**2)

        weight = np.exp(-new_error_sq*self.beta/2)

        if np.random.uniform() < weight:
            keep = True

            self.delta_error_squared = new_error_squared-self.error_squared
            if np.abs(self.delta_error_squared) <= self.exit_tol:
                converged = True
                self.converged = converged

        else:
            keep = False

        return keep, converged

    # ----------------------------------------------------------------------------------------------


