import numpy as np
from scipy import optimize


def find_latency_coding_lif_params(a_min=50, a_max=100, precision=10, tau=20):
    du = slice(0.01, 1, 0.02)
    dv = slice(0.01, 1, 0.02)
    k_vth = slice(1, 20, 0.5)

    params = (a_min, a_max, precision, tau)
    resbrute = optimize.brute(func=latency_coding_cost,
                              ranges=(du, dv, k_vth),
                              args=params,
                              full_output=True)

    print("\nGlobal minimum and function value at this min:", resbrute[0],
          resbrute[1])
    # print("\n Indices of zero costs: ", np.argwhere(resbrute[3] == 0))
    return resbrute[0]


def latency_coding_cost(z, *params):
    du, dv, k_vth = z
    a_min, a_max, precision, num_steps = params

    vth = k_vth * a_max

    cost = 0

    u0_vals = np.arange(precision, a_max+precision, precision)
    v0_vals = u0_vals
    n_sp_gen_u0 = len(u0_vals[u0_vals >= a_min])

    spikes, _, _ = CUBA_u_v_dyn(du, dv, vth, u0_vals.copy(), v0_vals.copy(),
                                num_steps)

    spike_times = np.where(spikes.any(axis=1), spikes.argmax(axis=1), -1) + 1

    below_a_min_spike_count = np.count_nonzero(spike_times[u0_vals < a_min])
    cost = cost + 1000 * below_a_min_spike_count

    above_a_min_spike_count = np.count_nonzero(spike_times[u0_vals >= a_min])
    cost = cost + 300 * (n_sp_gen_u0 - above_a_min_spike_count)

    unique_s_ts = len(np.unique(spike_times[u0_vals >= a_min]))
    cost = cost + 100 * (n_sp_gen_u0 - unique_s_ts)

    last_spike_ts = spike_times[u0_vals >= a_min][0]
    cost = cost + last_spike_ts

    return cost


def CUBA_u_v_dyn(du, dv, vth, u0, v0, num_steps):
    u = u0.astype('float')
    v = v0.astype('float')
    spikes = np.zeros(shape=(len(u0), num_steps))
    u_data = np.zeros(shape=(len(u0), num_steps))
    v_data = np.zeros(shape=(len(u0), num_steps))
    for ts in range(num_steps):
        u_data[:, ts] = u[:]
        v_data[:, ts] = v[:]
        u[:] = u * (1 - du)
        v[:] = v * (1 - dv) + u
        s_out = v >= vth
        spikes[:, ts] = s_out
        v[s_out] = 0  # Reset voltage to 0

    return spikes, u_data, v_data
