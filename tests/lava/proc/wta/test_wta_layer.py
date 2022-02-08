# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

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

from lava.proc.wta.process import WTALayer
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi1SimCfg


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


class TestWTALayerModels(unittest.TestCase):
    def test_wta_layer_constructor(self):
        """Test WTALayeer process constructor with default arguments"""
        wta_layer = WTALayer()

        self.assertIsInstance(wta_layer, WTALayer)

    def test_wta_proc_compiled_without_error(self):
        """Check if Monitor Proc is compiled without an error"""
        wta_layer = WTALayer()
        c = Compiler()
        # Compiling should run without error
        c.compile(wta_layer, Loihi1SimCfg())

    def test_wta_layer_run_without_error(self):
        """Test if the WTALayer proc run without any input"""
        num_steps = 4
        wta_layer = WTALayer()

        # Should run without error (not doing anything)
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi1SimCfg()

        wta_layer.run(condition=rcnd, run_cfg=rcfg)
        wta_layer.stop()

    def test_wta_layer_output_is_correct_with_input(self):
        """ Tests if WTA-layer works as excepted, i.e. the neuron that
        receive highest input activation a_in, will spike first and inhibit
        all its competitor neurons. This test uses only 4 LIF neurons with
        inputs [10,20,30,40] and we expect only the 4th neuron to spike"""

        # Setup input activation range to [20, 40] with precision 10
        num_steps = 10  # same as max_num_steps for latency coding
        a_min = 20
        a_max = 40
        precision = 10
        k_tau = 1

        # Actual range will include values below a_min too
        # Number of different input activation (current) values defines also
        # the shape of the layer, i.e. how many neurons will be in the layer
        current_inputs = np.arange(precision, a_max + precision, precision)
        shape = current_inputs.shape

        # Setup the WTA layer with provided parameters and constraints
        wta_layer = WTALayer(shape=shape,
                             a_min=a_min,
                             a_max=a_max,
                             precision=precision,
                             max_num_steps=num_steps,
                             k_tau=k_tau)

        # The input process to send these a_in values (current_inputs)
        act_input = VecSendProcess(shape=shape,
                                   num_steps=num_steps,
                                   vec_to_send=current_inputs)

        # Connection activation input process to the a_in of WTALayer process
        act_input.s_out.connect(wta_layer.a_in)

        # Create monitor and probe the output spike of the WTALayer
        monitor = Monitor()
        monitor.probe(wta_layer.s_out, num_steps=num_steps)

        # Config and run this layer
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi1SimCfg()

        wta_layer.run(condition=rcnd, run_cfg=rcfg)

        # Fetch and construct the monitored data with get_data(..) method
        data = monitor.get_data()

        # Access the collected data with the names of monitor proc and var
        probe_data = data[wta_layer.name][wta_layer.s_out.name]

        wta_layer.stop()

        # Validate that only the neuron (4th neuron) which received the highest
        # activation/current (a_in=40) will emit spike(s), while other
        # neurons should be silent
        self.assertTrue(((np.count_nonzero(probe_data, axis=0) > 0) ==
                        [False, False, False, True]).all())

        
if __name__ == '__main__':
    unittest.main()
