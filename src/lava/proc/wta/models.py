# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements
from lava.proc.wta.process import WTALayer
from lava.utils.latency_coding import find_latency_coding_lif_params


@implements(proc=WTALayer, protocol=LoihiProtocol)
class SubWTALayerModel(AbstractSubProcessModel):
    """
    This SubProcessModel of the WTALayer process. It first instantiates the
    child LIF and Dense Processes. Input and output of the WTALayer process
    are those of child LIF process. Dense process only implements the lateral
    inhibitory connections.
    """
    def __init__(self, proc):
        """Builds sub Process structure of the Process."""

        # Get parameters to instantiate child processes from the init_args of
        # WTALayer process
        shape = proc.init_args.get("shape")
        a_min = proc.init_args.get("a_min")
        a_max = proc.init_args.get("a_max")
        precision = proc.init_args.get("precision")
        max_num_steps = proc.init_args.get("max_num_steps")
        k_tau = proc.init_args.get("k_tau")

        # Find the optimized LIF parameters for latency coding based on the
        # given constraints by user
        opt_params = find_latency_coding_lif_params(a_min=a_min,
                                                    a_max=a_max,
                                                    precision=precision,
                                                    max_num_steps=max_num_steps,
                                                    k_tau=k_tau)
        # Unpack these params
        du, dv, vth = opt_params

        # Lateral inhibition weight matrix is 2D tensors with zeros on
        # diagonal, and negative weights everywhere else; so that each neuron
        # inhibits all other neurons in the population, if it spikes first.
        shape_weights = (shape[0], shape[0])
        weights = -a_max * (np.ones(shape=shape_weights) - np.eye(shape[0])) \
            .astype(int)

        # shape is a 2D vec (shape of weight mat)
        self.dense = Dense(shape=shape_weights, weights=weights)

        # shape is a 1D vec
        self.lif = LIF(shape=shape, du=du, dv=dv, vth=vth)

        # connect Parent in port to child Dense in port
        proc.in_ports.a_in.connect(self.lif.in_ports.a_in)
        # connect child LIF out port to parent out port
        self.lif.out_ports.s_out.connect(proc.out_ports.s_out)

        # connect LIF Proc out port to Dense Proc in port as lateral inhibition
        self.lif.out_ports.s_out.connect(self.dense.in_ports.s_in)
        # connect Dense Proc out port to LIF Proc in port
        self.dense.out_ports.a_out.connect(self.lif.in_ports.a_in)

        # Exposes the variables of the LIF and Dense child Processes  to the
        # WTALayer parent Process
        proc.vars.u.alias(self.lif.vars.u)
        proc.vars.v.alias(self.lif.vars.v)
        proc.vars.bias.alias(self.lif.vars.bias)
        proc.vars.du.alias(self.lif.vars.du)
        proc.vars.dv.alias(self.lif.vars.dv)
        proc.vars.vth.alias(self.lif.vars.vth)
        proc.vars.weights.alias(self.dense.vars.weights)
