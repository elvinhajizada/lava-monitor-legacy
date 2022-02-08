# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class WTALayer(AbstractProcess):
    """Winner-take-all layer (WTALayer) Process based on latency coding.

    This is a hierarchical process based on LIF neural and Dense connection
    processes. It adopts latency coding of synaptic input in LIF neurons,
    to map the input current to the latency of the spike emitted by the
    neuron. Dense connection is the deployed with inhibitory weights as
    the lateral inhibition on top of these LIF neurons.

    Some critical parameters (du, dv, vth) of the LIF neurons are set by a grid
    search algorithm through utility function find_latency_coding_lif_params().
    Other parameters of LIF and Dense process, e.g. bias, bias, bias_exp,
    weights etc. are set automatically by the process.

    User only need to define the shape of the LIF neuron population and
    latency coding parameters and constraints.

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
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        shape = kwargs.get("shape", (5,))
        if len(shape) != 1:
            raise AssertionError("Dense Process 'shape' expected a 1D tensor.")

        a_min = kwargs.get("a_min", 20)
        a_max = kwargs.get("a_max", 40)
        precision = kwargs.get("precision", 10)
        max_num_steps = kwargs.get("max_num_steps", 10)
        k_tau = kwargs.get("k_tau", 1)

        shape_weights = (shape[0], shape[0])

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        # Dense connection var(s)
        self.weights = Var(shape=shape_weights, init=0)

        # LIF var(s)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.bias = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=0)
        self.dv = Var(shape=(1,), init=0)
        self.vth = Var(shape=(1,), init=0)
