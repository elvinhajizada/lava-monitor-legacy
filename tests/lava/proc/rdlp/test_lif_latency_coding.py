import unittest
import numpy as np

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.monitor.process import Monitor
from lava.proc.prototypes.process import Prototypes
from lava.proc.lif.process import LIF
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi1SimCfg
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize


class VecSendProcess(AbstractProcess):
    """
    Process of a user-defined shape that sends an arbitrary vector

    Parameters
    ----------
    shape: tuple, shape of the process
    vec_to_send: np.ndarray, vector of spike values to send
    send_at_times: np.ndarray, vector bools. Send the `vec_to_send` at times
    when there is a True
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", (1,))
        vec_to_send = kwargs.pop("vec_to_send")
        num_steps = kwargs.pop("num_steps", 1)
        self.shape = shape
        self.num_steps = num_steps
        self.vec_to_send = Var(shape=shape, init=vec_to_send)
        self.s_out = OutPort(shape=shape)


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.current_ts == 1:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


class TestLatencyCodingLIF(unittest.TestCase):
    def test_something(self):

        num_steps = 40
        a_max = 100
        a_min = 50
        precision = 10

        n_neurons = int(a_max/precision)
        shape = (n_neurons,)
        current_inputs = np.arange(precision, a_max+precision, precision)

        du, dv, k_vth = find_latency_coding_lif_params(a_min=a_min,
                                                       a_max=a_max,
                                                       precision=precision,
                                                       tau=num_steps)

        # du, dv, k_vth = (0.01, 0.06, 5.5)

        vth = k_vth * a_max

        lif = LIF(shape=shape,
                  du=du,
                  dv=dv,
                  bias=0,
                  bias_exp=0,
                  vth=vth)

        act_input = VecSendProcess(shape=shape,
                                   num_steps=num_steps,
                                   vec_to_send=current_inputs)

        monitor = Monitor()
        monitor_u = Monitor()
        monitor_v = Monitor()

        act_input.s_out.connect(lif.a_in)

        monitor.probe(lif.s_out, num_steps=num_steps)
        monitor_u.probe(lif.u, num_steps=num_steps)
        # monitor_v.probe(lif.v, num_steps=num_steps)

        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi1SimCfg()

        lif.run(condition=rcnd, run_cfg=rcfg)

        # Fetch and construct the monitored data with get_data(..) method
        data = monitor.get_data()
        data_u = monitor_u.get_data()
        # data_v = monitor_v.get_data()

        # Access the collected data with the names of monitor proc and var
        probe_data = data[lif.name][lif.s_out.name]
        u_data = data_u[lif.name][lif.u.name]
        # v_data = data_v[lif.name][lif.v.name]

        lif.stop()

        # print(probe_data.T)

        plt.imshow(probe_data.T, cmap=plt.cm.gray)
        plt.show()

        spikes, u_sim, v_sim = CUBA_u_v_dyn(du, dv, vth,
                                            current_inputs.copy(),
                                            current_inputs.copy(),
                                            num_steps)

        spike_times = np.where(spikes.any(axis=1), spikes.argmax(axis=1),
                               -1)

        plt.imshow(spikes, cmap=plt.cm.gray)
        plt.show()

        print(u_sim[14, :])
        print(u_data[:, 14])
        # y0 = [75, 75]
        # t = np.arange(0, num_steps)
        # # sol = odeint(CUBA, y0, t, args=(du, dv))
        #
        # plt.plot(t, sol[:, 0], 'b', label='u(t)')
        # plt.plot(t, u_data[:, 14], 'b-', label='u_lif(t)')
        # # plt.plot(t, sol[:, 1], 'g', label='v(t)')
        # # plt.plot(t, v_data[:, 14], 'g-', label='v_lif(t)')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        # plt.show()
        # print(sol[:, 0])

    def test_latency_code_param_optimization(self):
        res = find_latency_coding_lif_params(a_min=50,
                                             a_max=100,
                                             precision=10,
                                             tau=40)


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


def CUBA(y, t, du, dv):
    u, v = y
    dydt = [-du*u, -dv*v + u]
    return dydt


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


if __name__ == '__main__':
    unittest.main()
