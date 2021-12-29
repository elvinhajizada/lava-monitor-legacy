import unittest
import numpy as np

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires
from lava.proc.monitor.process import Monitor
from lava.proc.lif.process import LIF
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.utils.latency_coding import find_latency_coding_lif_params, \
    CUBA_u_v_dyn


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
        Send `spikes_to_send` if current time-step is 1
        """
        if self.current_ts == 1:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


class TestLatencyCodingLIF(unittest.TestCase):
    def test_latency_code_param_optimization(self):
        """
        Tests the activation-to-latency coding feature of LIF neurons. More
        specifically it tests the util func find_latency_coding_lif_params().
        This Function that search for the optimal parameters of LIF neuron model
        tha can convert its input activation values "a_in" into the latency
        of its output spike.
        """

        # Define constraints for activity-to-latency coding...

        # Max number of time steps that simulation will be run for, also the
        # number of max time steps allowed for latency coding
        num_steps = 40
        a_max = 100  # Upper bound of activation range that we want to sample
        a_min = 50   # Lower bound of activation range that we want to sample
        precision = 5  # Precision of sampling in the  range of the activation
        k_tau = 2

        # Calculate number of neurons necessary to simulate all sampled values
        # of activation by stimulating these neurons correspondingly
        n_neurons = int(a_max/precision)
        shape = (n_neurons,)  # the shape of neural population

        # Define current (u) inputs, which are sampled from the provided
        # range with given precision. These current (u) values will be fed to
        # the corresponding neurons
        current_inputs = np.arange(precision, a_max+precision, precision)

        # Run the grid search for (du, dv, vth) to find optimal parameters
        # for activation-to-latency coding
        du, dv, vth = find_latency_coding_lif_params(a_min=a_min,
                                                     a_max=a_max,
                                                     precision=precision,
                                                     max_num_steps=num_steps,
                                                     k_tau=k_tau)

        # Create a LIF neuron population with these optimized parameters.
        # These neurons will receives different a_in (activations) and will
        # convert it to the latency of their spikes
        lif = LIF(shape=shape,
                  du=du,
                  dv=dv,
                  bias=0,
                  bias_exp=0,
                  vth=vth)

        # The input process to send these a_in values using above created array
        act_input = VecSendProcess(shape=shape,
                                   num_steps=num_steps,
                                   vec_to_send=current_inputs)

        # Create monitor and probe the output spike of these neurons
        monitor = Monitor()
        monitor.probe(lif.s_out, num_steps=num_steps)

        # Connection activation input process to the a_in of LIF process
        act_input.s_out.connect(lif.a_in)

        # Use standard Loihi1SimCfg
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi1SimCfg()

        lif.run(condition=rcnd, run_cfg=rcfg)

        # Fetch and construct the monitored data with get_data(..) method
        data = monitor.get_data()

        # Access the collected data with the names of monitor proc and var
        probe_data = data[lif.name][lif.s_out.name]

        lif.stop()

        # Get the spike times of the first spike (latency coding) for each input
        # activation value
        spike_times = np.where(probe_data.T.any(axis=1),
                               probe_data.T.argmax(axis=1), -1)

        # Number of spike generating u0 values, i.e. the number activation (
        # current) values that should generate a spike
        n_sp_gen_u0 = len(current_inputs[current_inputs >= a_min])

        # Count the number of activation values that incorrectly DID NOT
        # generate a spike. This value should be zero
        below_a_min_spike_count = np.count_nonzero(
                spike_times[current_inputs < a_min] != -1)
        self.assertEqual(below_a_min_spike_count, 0,
                         "Some below a_min inputs generated a spike")

        # Count the number of activation values that incorrectly DID NOT
        # generate a spike This value should be equal to the number of spike
        # generating u0 values, i.e. "n_sp_gen_u0"
        above_a_min_spike_count = np.count_nonzero(
                spike_times[current_inputs >= a_min] != -1)
        self.assertEqual(above_a_min_spike_count, n_sp_gen_u0,
                         "Not all inputs in the range have generated spike")

        # Count the number of unique latencies for those activations that should
        # have generated a spike. This value should be equal to the number of
        # spike generating u0 values, i.e. "n_sp_gen_u0"
        unique_s_ts = len(np.unique(spike_times[current_inputs >= a_min]))
        self.assertEqual(unique_s_ts, n_sp_gen_u0,
                         "Not all generated spikes have unique latency")

    def test_lif_num_sim_match_lava_lif(self):
        """ Tests CUBA_u_v_dyn() function that simulates the current, voltage
        and spiking dynamics of LIF (CUBA) neuron model. This simple
        simulation is validated to produce same results as Lava LIF process.
        This test is responsible for the validation of spike times"""

        # Setup
        num_steps = 40
        a_max = 100
        precision = 5

        n_neurons = int(a_max / precision)
        shape = (n_neurons,)
        current_inputs = np.arange(precision, a_max + precision, precision)

        # The optimal values found in the test_latency_code_param_optimization()
        du, dv, vth = [0.0125, 0.0325, 800]

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

        act_input.s_out.connect(lif.a_in)

        monitor.probe(lif.s_out, num_steps=num_steps)

        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi1SimCfg()

        lif.run(condition=rcnd, run_cfg=rcfg)

        # Fetch and construct the monitored data with get_data(..) method
        data = monitor.get_data()

        # Access the collected data with the names of monitor proc and var
        probe_data = data[lif.name][lif.s_out.name]

        lif.stop()

        # Get the spike data from the CUBA_u_v_dyn() simulation with the same
        # parameters
        spikes, _, _ = CUBA_u_v_dyn(du, dv, vth,
                                    current_inputs.copy(),
                                    current_inputs.copy(),
                                    num_steps)

        # Validate that they are equal
        self.assertTrue(np.array(spikes == probe_data.T).all())

        # Comment in to see the simple spike raster plots
        # plt.imshow(spikes, cmap=plt.cm.gray)
        # plt.show()
        #
        # plt.imshow(probe_data.T, cmap=plt.cm.gray)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
