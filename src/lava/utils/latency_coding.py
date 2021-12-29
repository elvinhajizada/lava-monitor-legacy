# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from scipy import optimize

"""
This utility file contains necessary functions to find optimal parameter for 
a LIF neuron model that can convert its input activation values "a_in" into 
latency of its output spike. 

Latency coding can utilized together with a winner-take-all mechanism to 
identify the neuron with highest activation input. Note that this "a_in" is 
accumulated input after going through a connection/synapse, thus, possibly
multiplied by a weight matrix.

This utility functions search for such a collection of parameters for LIF 
neurons that the user-defined activity range is samples uniquely by spiking 
latency based on the precision also provided by user. The activation values 
below the lower bound of the range will not produce any spikes with the found
parameters. It uses the grid search over a parameter space of "du", 
"dv" and "vth" from LIF neuron model.

The algorithm also parametrizes the search grid based on max_num_steps of 
latency coding window that is allowed by user. However, it is not guaranteed 
that the optimal parameters found by the algorithm given the user-defined 
time constraints, will provide the requested precision in sampling the 
activation range. Theoretically any range can be sampled with an arbitrary 
precision given that latency coding window is indefinitely long. However, 
this search algorithm tries to find the most optimal solution given the 
constraints.

Optimality is defined in terms of a cost functions, which has the following 
rank of penalty constants for each (ranked 
from strongest to weakest penalty):
1. Spiking below the lower bound of user-provided activation range
2. Sampling precision violations 
3. The time length of latency coding window in number of time steps
"""


def find_latency_coding_lif_params(a_min=50, a_max=100, precision=10,
                                   max_num_steps=20, k_tau=1):
    """
    Function that search for the optimal parameters of LIF neuron model tha can
    convert its input activation values "a_in" into the latency of its output
    spike.

    Parameters
    ----------
    a_min : Lower bound of the activation range to be sampled
    a_max : Upper bound of the activation range to be sampled
    precision : Precision of the sampling in the given range of the activation
    max_num_steps : Number of max time steps allowed for latency coding
    k_tau : Scalar term that defines the maximum time constant of current and
    voltage decay of the LIF neuron in relation to max_num_steps:
    tau_v = k_tau * max_num_steps
    tau_u = k_tau * max_num_steps

    Returns
    -------
    optimized_params : The tuple of optimal parameters for the LIF neuron
    model to sample given range of the activation with requested precision
    under the provided time  constraints. The results is in the form (du,dv,
    vth), where du is the inverse of decay time-constant for current decay;
    dv is the inverse of decay time-constant for voltage decay; vth is  the
    voltage threshold
    """
    # Define grid of the search by setting the slices for each parameter

    # Lower bound of du and dv (hence upper bound of the respective taus) are
    # defined based on maximum allowed time steps for latency coding window
    # and how fast u and v should decay compared to this window (i.e. through
    # k_tau parameter)
    du = slice(1/(k_tau*max_num_steps), 1, 0.02)
    dv = slice(1/(k_tau*max_num_steps), 1, 0.02)

    # vth slice is defined based on a_max
    vth = slice(0.75*a_max, 10*a_max, 0.25*a_max)

    # Set the other variable that will stay constant
    params = (a_min, a_max, precision, max_num_steps)

    # Run the grid search to minimize the cost produced by the
    # latency_coding_cost() function over the defined grid of parameters
    grid_search_results = optimize.brute(func=latency_coding_cost,
                                         ranges=(du, dv, vth),
                                         args=params,
                                         full_output=True,
                                         finish=None)

    # The found optimal params as a tuple: (du,dv,vth)
    optimized_params = grid_search_results[0]

    # The cost function value at this minimum
    min_cost_value = grid_search_results[1]

    # Print the results
    print("\nGlobal minimum and function value at this min:",
          optimized_params, min_cost_value)

    return optimized_params


def latency_coding_cost(grid_params, *fixed_params):
    """
    Cost function to be minimized while searching for activation-to-latency
    coding LIF neuron parameters.
    Cost is calculated as following:

    cost = a*N_{FP}+b*N_{FN}+c*N_{OL}+t_{LS}

    a,b,c,d : scalar
    N_{FP} : number of false positives, i.e. the activation values that
    should not have generated spike (i.e. below a_min) but did
    N_{FN} : number of false negatives, i.e. the activation values that
    should have generated spike (i.e. above a_min) but did not
    N_{OL} : number of latency overlaps, i.e. number of activation values in
    the range that should have generated different latency spikes, but did
    with the same latency
    t_{LS} : the latency of the spike generated by a_min

    Parameters
    ----------
    grid_params : parameters with ranges to be searched
    fixed_params : fixed parameters

    Returns
    -------
    cost : cost function value for given set of parameters
    """

    # Unpack both grid and fixed params
    du, dv, vth = grid_params
    a_min, a_max, precision, num_steps = fixed_params

    # Initialize the cost to zero
    cost = 0

    # Define the range of input activation values (i.e. initial current)
    # based on the user defined precision and upper bound of the activation
    # range. This range does not finish at a_min, as we want to test those
    # values below a_min to validate that they don't generate any spike at all
    u0_vals = np.arange(precision, a_max+precision, precision)

    # Initialize voltage values to the intial current values in line with LIF
    # process model implementation
    v0_vals = u0_vals

    # Number of spike generating u0 values, i.e. the number activation (
    # current) values that should generate a spike
    n_sp_gen_u0 = len(u0_vals[u0_vals >= a_min])

    # run the simulation of CUBA LIF model for u,v and spike dynamics using
    # the provided helper function to get the spike data
    spikes, _, _ = CUBA_u_v_dyn(du, dv, vth, u0_vals.copy(), v0_vals.copy(),
                                num_steps)

    # Get the spike times of the first spike (latency coding) for each input
    # activation value
    spike_times = np.where(spikes.any(axis=1), spikes.argmax(axis=1), -1)

    # Count the number of activation values that incorrectly generated a spike
    # i.e. those below a_min and add this to the cost with highest penalty
    below_a_min_spike_count = np.count_nonzero(spike_times[u0_vals < a_min]
                                               != -1)
    cost = cost + 1000 * below_a_min_spike_count

    # Count the number of activation values that incorrectly DID NOT generate
    # a spike and add it to the cost with the second highest penalty term
    above_a_min_spike_count = np.count_nonzero(spike_times[u0_vals >= a_min]
                                               != -1)
    cost = cost + 300 * (n_sp_gen_u0 - above_a_min_spike_count)

    # Count the number of unique latencies for those activations that should
    # have generated a spike. Any divergence from the optimal count
    # (n_sp_gen_u0) is added to the cost with the third highest penalty term
    unique_s_ts = len(np.unique(spike_times[u0_vals >= a_min]))
    cost = cost + 100 * (n_sp_gen_u0 - unique_s_ts)

    # Finally when and if all of the other components of the cost is zero,
    # the deciding factor for the optimality of the parameter is decided
    # based upon the latency of the spike for a_min, i.e. the last spike that
    # can be outputted. This way the latency coding window is shortened as
    # much as possible. For this purpose, latency of the spike with highest
    # latency is the last (and smallest) penalty term
    last_spike_ts = spike_times[u0_vals >= a_min][0]
    cost = cost + last_spike_ts

    return cost


def CUBA_u_v_dyn(du, dv, vth, u0, v0, num_steps):
    """
    A function that simulates the current, voltage and spiking dynamics of
    LIF (CUBA) neuron model for num_steps given the initial values of the
    current and voltage.

    Note: This simulation is based on LIF process model with tag(
    'floating_pt'). It is developed to serve as helper function for
    latency_coding_cost() function

    Parameters
    ----------
    du : Inverse of decay time-constant for current decay
    dv : Inverse of decay time-constant for voltage decay
    vth : Neuron threshold voltage, exceeding which, the neuron will spike.
    u0 : Initial value of the current at the beginning of simulation
    v0 : Initial value of the voltage at the beginning of simulation
    num_steps : num of time steps that simulation will be run

    Returns
    -------
    spikes : binary spike matrix as output of simulation
    u_data : current data matrix as output of simulation
    v_data : voltage data matrix as output of simulation
    """
    u = u0.astype('float')
    v = v0.astype('float')

    # Data collections
    spikes = np.zeros(shape=(len(u0), num_steps))
    u_data = np.zeros(shape=(len(u0), num_steps))
    v_data = np.zeros(shape=(len(u0), num_steps))

    # run the simulation for num_steps
    for ts in range(1, num_steps):
        u_data[:, ts] = u[:]
        v_data[:, ts] = v[:]
        u[:] = u * (1 - du)
        v[:] = v * (1 - dv) + u
        s_out = v >= vth
        spikes[:, ts] = s_out
        v[s_out] = 0  # Reset voltage to 0

    return spikes, u_data, v_data
