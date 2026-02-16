import numpy as np
import pints
from numba import jit
from scipy.integrate import odeint
from scipy.stats import moment

import math

# from pints import ToyModel


class SmolModel(pints.ForwardModel):

    r"""
    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_population_size : float
        Sets the initial population size :math:`p_0`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Population_growth
    """

    def __init__(self, N, n0):
        super(SmolModel, self).__init__()

        # Check initial values
        if N is None:
            self._n0 = np.zeros((100))
            self._n0[0] = 500
            # print('Init 1')
        #     # self._y0 = np.array([38, 1, 0])
        else:
            self._n0 = np.array((100))
            self._n0[0] = N
            # print('Init 2')

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 3

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        # IN time change this to N + 2 parameters (one b for each size)
        return 4


    @staticmethod
    @jit
    def _rhs(n0,t_curr,b,p,s,N):
        # Fix the maximal cluster size
        N_max = 100

        dn_dt = np.zeros(N_max)
        for i in range(0,N_max):
            # 1st sum of coagulation term calculation is coag_gain
            coag_gain = 0
            # 1st sum of coagulation term calculation is coag_loss
            coag_loss = 0

            prolif_gain = 0
            prolif_loss = 0

            splitting_loss = 0
            splitting_gain = 0

            if i == 0:
                coag_gain = 0
                for j in range(0,N_max-i-1):
                    coag_loss += b*1*n0[i]*n0[j]
                
                prolif_gain = 0
                prolif_loss = p*(i+1)*n0[i]

                splitting_gain = s * (2) * n0[1]
                for l in range(i+1,N_max):
                        splitting_gain += s * (l+1) * n0[l]


                

            elif i < N_max-1:
                for j in range(0,i):
                    coag_gain += b*1*n0[i-j-1]*n0[j]
                
                for j in range(0,N_max-i-1):
                    coag_loss += b*1*n0[i]*n0[j]

                prolif_gain = p*(i)*n0[i-1]
                prolif_loss = p*(i+1)*n0[i]  

                splitting_loss = s * (i+1) * n0[i]
                splitting_gain = s * (i+2) * n0[i+1]            


            elif i == N_max - 1:
                for j in range(0,i):
                    coag_gain += b*1*n0[i-j-1]*n0[j]
                
                coag_loss = 0

                prolif_gain = p*(i)*n0[i-1]
                prolif_loss = 0

                splitting_loss = s * (N_max) * n0[N_max-1]


            coagulation = (1/2)*coag_gain - coag_loss

            proliferation = prolif_gain - prolif_loss
                

            splitting = splitting_gain - splitting_loss


            dn_dt[i] = coagulation + proliferation + splitting


        return dn_dt



    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        b, p, s, N = parameters
        n0 = np.array(self._n0)
        n0[0] = N
        # n = odeint(self._rhs, n0, times, (b,N))
        # print('Times are:', times)
        # print('n0 is', n0)
        if times[0] != 0:
            time_gap = int(times[1] - times[0])
            previous_times = np.linspace(0, int(times[0]), int((int(times[0])-0)/time_gap) + 1)
            # print('Previous times', previous_times)
            # print('Inputs', n0, previous_times, b, N)
            n_previous_times = odeint(SmolModel._rhs, n0, previous_times, (b,p,s,N))
            n_input = n_previous_times[-1,:]
            # print('N input', n_input)
            # print('Length', len(n_input))
        else:
            n_input = n0
        n = odeint(SmolModel._rhs, n_input, times, (b,p,s,N))


        out_array = np.zeros((n.shape[0],3))

        for i in range(np.shape(n)[0]):
            model_slice = n[i,:]


            s0 = sum(model_slice)
            s1 = (1/s0)*(sum((j + 1) * phi_j for j, phi_j in enumerate(model_slice)))
            s2 = (1/s0)*(sum((j + 1)**2 * phi_j for j, phi_j in enumerate(model_slice)))
            s2_power_half = s2**(1/2)

            out_array[i,0] = s0
            out_array[i,1] = s1
            out_array[i,2] = s2


        return out_array
        # return n

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """

        return np.array([0.0004, 0.01, 0.0001, 100])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """

        return np.linspace(0, 97, 97000)