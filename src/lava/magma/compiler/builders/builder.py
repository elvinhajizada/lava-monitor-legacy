# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
import typing as ty

import numpy as np
from dataclasses import dataclass

from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.message_infrastructure.message_infrastructure_interface\
    import MessageInfrastructureInterface
from lava.magma.runtime.runtime_services.enums import LoihiVersion
from lava.magma.runtime.runtime_services.runtime_service import (
    AbstractRuntimeService,
    NxSdkRuntimeService
)

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.runtime.runtime import Runtime

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.compiler.builders.interfaces import (
    AbstractProcessBuilder,
    AbstractRuntimeServiceBuilder,
    AbstractChannelBuilder
)
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.compiler.utils import VarInitializer, PortInitializer, \
    VarPortInitializer
from lava.magma.core.model.py.ports import (
    AbstractPyIOPort,
    PyInPort,
    PyOutPort,
    PyRefPort,
    PyVarPort
)
from lava.magma.compiler.channels.interfaces import AbstractCspPort, Channel, \
    ChannelType


class _AbstractProcessBuilder(AbstractProcessBuilder):
    """A _AbstractProcessBuilder instantiates and initializes
    an AbstractProcessModel but is not meant to be used
    directly but inherited from"""

    def __init__(
            self, proc_model: ty.Type[AbstractProcessModel],
            model_id: int):
        super(_AbstractProcessBuilder, self).__init__(
            proc_model=proc_model,
            model_id=model_id
        )

    @property
    def proc_model(self) -> ty.Type[AbstractProcessModel]:
        return self._proc_model

    # ToDo: Perhaps this should even be done in Compiler?
    def _check_members_exist(self, members: ty.Iterable, m_type: str):
        """Checks that ProcessModel has same members as Process.

        Parameters
        ----------
        members : ty.Iterable

        m_type : str

        Raises
        ------
        AssertionError
            Process and ProcessModel name should match
        """
        proc_name = self.proc_model.implements_process.__name__
        proc_model_name = self.proc_model.__name__
        # If proc_model have the member with the same name continue;
        # Otherwise check if parent_list_name is not None and proc_model has
        # a member with this parent_list_name, if so continue; otherwise
        # raise the error
        for m in members:
            if hasattr(self.proc_model, m.name):
                continue
            if m.parent_list_name is not None:
                if hasattr(self.proc_model, m.parent_list_name):
                    continue

            raise AssertionError(
                    "Both Process '{}' and ProcessModel '{}' are "
                    "expected to have {} named '{}'.".format(
                            proc_name, proc_model_name, m_type,
                            m.name
                    )
            )

    @staticmethod
    def _check_not_assigned_yet(
        collection: dict, keys: ty.Iterable[str], m_type: str
    ):
        """Checks that collection dictionary not already contain given keys
        to prevent overwriting of existing elements.

        Parameters
        ----------
        collection : dict

        keys : ty.Iterable[str]

        m_type : str


        Raises
        ------
        AssertionError

        """
        for key in keys:
            if key in collection:
                raise AssertionError(
                    f"Member '{key}' already found in {m_type}."
                )

    # ToDo: Also check that Vars are initializable with var.value provided
    def set_variables(self, variables: ty.List[VarInitializer]):
        """Appends the given list of variables to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        variables : ty.List[VarInitializer]

        """
        self._check_members_exist(variables, "Var")
        new_vars = {v.name: v for v in variables}
        self._check_not_assigned_yet(self.vars, new_vars.keys(), "vars")
        self.vars.update(new_vars)


class PyProcessBuilder(_AbstractProcessBuilder):
    """A PyProcessBuilder instantiates and initializes a PyProcessModel.

    The compiler creates a PyProcessBuilder for each PyProcessModel. In turn,
    the runtime, loads a PyProcessBuilder onto a compute node where it builds
    the PyProcessModel and its associated ports.

    In order to build the PyProcessModel, the builder inspects all LavaType
    class variables of a PyProcessModel, creates the corresponding data type
    with the specified properties, the shape and the initial value provided by
    the Lava Var. In addition, the builder creates the required PyPort
    instances. Finally, the builder assigns both port and variable
    implementations to the PyProcModel.

    Once the PyProcessModel is built, it is the RuntimeService's job to
    connect channels to ports and start the process.

    Note: For unit testing it should be possible to build processes locally
    instead of on a remote node. For pure atomic unit testing a ProcessModel
    locally, PyInPorts and PyOutPorts must be fed manually with data.
    """

    def __init__(
            self, proc_model: ty.Type[AbstractPyProcessModel],
            model_id: int,
            proc_params: ty.Dict[str, ty.Any] = None):
        super(PyProcessBuilder, self).__init__(
            proc_model=proc_model,
            model_id=model_id
        )
        if not issubclass(proc_model, AbstractPyProcessModel):
            raise AssertionError("Is not a subclass of AbstractPyProcessModel")
        self.vars: ty.Dict[str, VarInitializer] = {}
        self.py_ports: ty.Dict[str, PortInitializer] = {}
        self.ref_ports: ty.Dict[str, PortInitializer] = {}
        self.var_ports: ty.Dict[str, VarPortInitializer] = {}
        self.csp_ports: ty.Dict[str, ty.List[AbstractCspPort]] = {}
        self.csp_rs_send_port: ty.Dict[str, CspSendPort] = {}
        self.csp_rs_recv_port: ty.Dict[str, CspRecvPort] = {}
        self.proc_params = proc_params

    @property
    def proc_model(self) -> ty.Type[AbstractPyProcessModel]:
        return self._proc_model

    def check_all_vars_and_ports_set(self):
        """Checks that Vars and PyPorts assigned from Process have a
        corresponding LavaPyType.

        Raises
        ------
        AssertionError
            No LavaPyType found in ProcModel
        """
        for attr_name in dir(self.proc_model):
            attr = getattr(self.proc_model, attr_name)
            # Add all the parent_list names into one list and later check if
            # attr_name is in his list, in addition to checking it in port
            # and var dicts
            all_parent_list_names = [value.parent_list_name for value in
                                     list(self.vars.values()) +
                                     list(self.var_ports.values()) +
                                     list(self.py_ports.values()) +
                                     list(self.ref_ports.values())]
            if isinstance(attr, LavaPyType):
                if (
                    attr_name not in self.vars
                    and attr_name not in self.py_ports
                    and attr_name not in self.ref_ports
                    and attr_name not in self.var_ports
                    and attr_name not in all_parent_list_names
                ):
                    raise AssertionError(
                        f"No LavaPyType '{attr_name}' found in ProcModel "
                        f"'{self.proc_model.__name__}'."
                    )

    def check_lava_py_types(self):
        """Checks correctness of LavaPyTypes.

        Any Py{In/Out/Ref}Ports must be strict sub-types of Py{In/Out/Ref}Ports.
        """
        # If parent_list_name is not None, i.e. port has a parent container,
        # then get LavaType of that container
        for name, port_init in self.py_ports.items():
            if port_init.parent_list_name is not None:
                lt = self._get_lava_type(port_init.parent_list_name)
            else:
                lt = self._get_lava_type(name)

            if not isinstance(lt.cls, type):
                raise AssertionError(
                    f"LavaPyType.cls for '{name}' is not a type in '"
                    f"{self.proc_model.__name__}'."
                )
            if port_init.port_type == "InPort":
                if not (issubclass(lt.cls, PyInPort) and lt.cls != PyInPort):
                    raise AssertionError(
                        f"LavaPyType for '{name}' must be a strict sub-type "
                        f"of PyInPort in '{self.proc_model.__name__}'."
                    )
            elif port_init.port_type == "OutPort":
                if not (issubclass(lt.cls, PyOutPort) and lt.cls != PyOutPort):
                    raise AssertionError(
                        f"LavaPyType for '{name}' must be a strict sub-type of "
                        f"PyOutPort in '{self.proc_model.__name__}'."
                    )
            elif port_init.port_type == "RefPort":
                if not (issubclass(lt.cls, PyRefPort) and lt.cls != PyRefPort):
                    raise AssertionError(
                        f"LavaPyType for '{name}' must be a strict sub-type of "
                        f"PyRefPort in '{self.proc_model.__name__}'."
                    )

    def set_py_ports(self, py_ports: ty.List[PortInitializer], check=True):
        """Appends the given list of PyPorts to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        py_ports : ty.List[PortInitializer]

        check : bool, optional
             , by default True
        """
        if check:
            self._check_members_exist(py_ports, "Port")
        new_ports = {p.name: p for p in py_ports}
        self._check_not_assigned_yet(self.py_ports, new_ports.keys(), "ports")
        self.py_ports.update(new_ports)

    def set_ref_ports(self, ref_ports: ty.List[PortInitializer]):
        """Appends the given list of RefPorts to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        ref_ports : ty.List[PortInitializer]
        """
        self._check_members_exist(ref_ports, "Port")
        new_ports = {p.name: p for p in ref_ports}
        self._check_not_assigned_yet(self.ref_ports, new_ports.keys(), "ports")
        self.ref_ports.update(new_ports)

    def set_var_ports(self, var_ports: ty.List[VarPortInitializer]):
        """Appends the given list of VarPorts to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        var_ports : ty.List[VarPortInitializer]
        """
        new_ports = {p.name: p for p in var_ports}
        self._check_not_assigned_yet(self.var_ports, new_ports.keys(), "ports")
        self.var_ports.update(new_ports)

    def set_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Appends the given list of CspPorts to the ProcessModel. Used by the
        runtime to configure csp ports during initialization (_build_channels).

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]


        Raises
        ------
        AssertionError
            PyProcessModel has no port of that name
        """
        new_ports = {}
        for p in csp_ports:
            new_ports.setdefault(p.name, []).extend(
                p if isinstance(p, list) else [p]
            )

        # Check that there's a PyPort for each new CspPort
        # Check also parent_list_names of all ports
        proc_name = self.proc_model.implements_process.__name__
        all_ports = {**self.ref_ports, **self.py_ports, **self.var_ports}
        for port_name in new_ports:
            if not hasattr(self.proc_model, port_name) and not hasattr(
                    self.proc_model, all_ports[port_name].parent_list_name):
                raise AssertionError("PyProcessModel '{}' has \
                    no port named '{}'.".format(proc_name, port_name))

            if port_name in self.csp_ports:
                self.csp_ports[port_name].extend(new_ports[port_name])
            else:
                self.csp_ports[port_name] = new_ports[port_name]

    def set_rs_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Set RS CSP Ports

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]

        """
        for port in csp_ports:
            if isinstance(port, CspSendPort):
                self.csp_rs_send_port.update({port.name: port})
            if isinstance(port, CspRecvPort):
                self.csp_rs_recv_port.update({port.name: port})

    def _get_lava_type(self, name: str) -> LavaPyType:
        return getattr(self.proc_model, name)

    # ToDo: Need to differentiate signed and unsigned variable precisions
    # TODO: (PP) Combine PyPort/RefPort/VarPort initialization
    # TODO: (PP) Find a cleaner way to find/address csp_send/csp_recv ports (in
    #  Ref/VarPort initialization)
    def build(self):
        """Builds a PyProcModel at runtime within Runtime.

        The Compiler initializes the PyProcBuilder with the ProcModel,
        VarInitializers and PortInitializers.
        The Runtime builds the channels and CSP ports between all ports,
        assigns them to builder.

        At deployment to a node, the Builder.build(..) gets executed
        resulting in the following:
          1. ProcModel gets instantiated
          2. Vars are initialized and assigned to ProcModel
          3. PyPorts are initialized (with CSP ports) and assigned to ProcModel

        Returns
        -------
        AbstractPyProcessModel


        Raises
        ------
        NotImplementedError
        """

        # Create the ProcessModel
        # Default value of pm.proc_params in ProcessModel is an empty dictionary
        # If a proc_params argument is provided in PyProcessBuilder,
        # this will be carried to ProcessModel
        pm = self.proc_model(self.proc_params)
        pm.model_id = self._model_id

        # Initialize PyPorts
        for name, p in self.py_ports.items():
            # Build PyPort
            # If parent_list_name is not None, i.e. port has a parent container,
            # then get LavaType of that container
            if p.parent_list_name is not None:
                lt = self._get_lava_type(p.parent_list_name)
            else:
                lt = self._get_lava_type(name)
            port_cls = ty.cast(ty.Type[AbstractPyIOPort], lt.cls)
            csp_ports = []
            if name in self.csp_ports:
                csp_ports = self.csp_ports[name]
                if not isinstance(csp_ports, list):
                    csp_ports = [csp_ports]

            # TODO (MR): This is probably just a temporary hack until the
            #  interface of PyOutPorts has been adjusted.
            if issubclass(port_cls, PyInPort):
                port = port_cls(csp_ports, pm, p.shape, lt.d_type,
                                p.transform_funcs)
            elif issubclass(port_cls, PyOutPort):
                port = port_cls(csp_ports, pm, p.shape, lt.d_type)
            else:
                raise AssertionError("port_cls must be of type PyInPort or "
                                     "PyOutPort")

            # Create dynamic PyPort attribute on ProcModel
            setattr(pm, name, port)
            # Create private attribute for port precision
            # setattr(pm, "_" + name + "_p", lt.precision)

        # Initialize RefPorts
        for name, p in self.ref_ports.items():
            # Build RefPort
            # If parent_list_name is not None, i.e. port has a parent container,
            # then get LavaType of that container
            if p.parent_list_name is not None:
                lt = self._get_lava_type(p.parent_list_name)
            else:
                lt = self._get_lava_type(name)

            port_cls = ty.cast(ty.Type[PyRefPort], lt.cls)
            csp_recv = None
            csp_send = None
            if name in self.csp_ports:
                csp_ports = self.csp_ports[name]
                csp_recv = csp_ports[0] if isinstance(
                    csp_ports[0], CspRecvPort) else csp_ports[1]
                csp_send = csp_ports[0] if isinstance(
                    csp_ports[0], CspSendPort) else csp_ports[1]

            port = port_cls(csp_send, csp_recv, pm, p.shape, lt.d_type,
                            p.transform_funcs)

            # Create dynamic RefPort attribute on ProcModel
            setattr(pm, name, port)

        # Initialize VarPorts
        for name, p in self.var_ports.items():
            # Build VarPort
            if p.port_cls is None:
                # VarPort is not connected
                continue
            port_cls = ty.cast(ty.Type[PyVarPort], p.port_cls)
            csp_recv = None
            csp_send = None
            if name in self.csp_ports:
                csp_ports = self.csp_ports[name]
                csp_recv = csp_ports[0] if isinstance(
                    csp_ports[0], CspRecvPort) else csp_ports[1]
                csp_send = csp_ports[0] if isinstance(
                    csp_ports[0], CspSendPort) else csp_ports[1]
            port = port_cls(
                p.var_name, csp_send, csp_recv, pm, p.shape, p.d_type,
                p.transform_funcs)

            # Create dynamic VarPort attribute on ProcModel
            setattr(pm, name, port)

        for port in self.csp_rs_recv_port.values():
            if "service_to_process" in port.name:
                pm.service_to_process = port
                continue

        for port in self.csp_rs_send_port.values():
            if "process_to_service" in port.name:
                pm.process_to_service = port
                continue

        # Initialize Vars
        for name, v in self.vars.items():
            # Build variable
            # If parent_list_name is not None, i.e. Var has a parent container,
            # then get LavaType of that container
            if v.parent_list_name is not None:
                lt = self._get_lava_type(v.parent_list_name)
            else:
                lt = self._get_lava_type(name)

            if issubclass(lt.cls, np.ndarray):
                var = lt.cls(v.shape, lt.d_type)
                var[:] = v.value
            elif issubclass(lt.cls, (int, float)):
                var = v.value
            else:
                raise NotImplementedError

            # Create dynamic variable attribute on ProcModel
            setattr(pm, name, var)
            # Create private attribute for variable precision
            setattr(pm, "_" + name + "_p", lt.precision)

            pm.var_id_to_var_map[v.var_id] = name

        return pm


class CProcessBuilder(_AbstractProcessBuilder):
    """C Process Builder"""

    pass


class NcProcessBuilder(_AbstractProcessBuilder):
    """NcProcessBuilder instantiates and initializes an NcProcessModel.

    The compiler creates a NcProcessBuilder for each NcProcessModel. In turn,
    the runtime, loads a NcProcessBuilder onto a compute node where it builds
    the NcProcessModel and its associated vars.

    In order to build the NcProcessModel, the builder inspects all LavaType
    class variables of a NcProcessModel, creates the corresponding data type
    with the specified properties, the shape and the initial value provided by
    the Lava Var. Finally, the builder assigns variable
    implementations to the NcProcModel."""
    def __init__(
            self, proc_model: ty.Type[AbstractNcProcessModel],
            model_id: int,
            proc_params: ty.Dict[str, ty.Any] = None):
        super(NcProcessBuilder, self).__init__(
            proc_model=proc_model,
            model_id=model_id
        )
        if not issubclass(proc_model, AbstractNcProcessModel):
            raise AssertionError("Is not a subclass of AbstractNcProcessModel")
        self.vars: ty.Dict[str, VarInitializer] = {}
        self.proc_params = proc_params

    def _get_lava_type(self, name: str) -> LavaNcType:
        return getattr(self.proc_model, name)

    def check_all_vars_set(self):
        """Checks that Vars assigned from Process have a
        corresponding LavaNcType.

        Raises
        ------
        AssertionError
            No LavaNcType found in ProcModel
        """
        for attr_name in dir(self.proc_model):
            attr = getattr(self.proc_model, attr_name)
            if isinstance(attr, LavaNcType):
                if (
                    attr_name not in self.vars
                ):
                    raise AssertionError(
                        f"No LavaNcType '{attr_name}' found in ProcModel "
                        f"'{self.proc_model.__name__}'."
                    )

    def set_rs_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        pass

    def build(self):
        """Builds a NcProcessModel at runtime within Runtime.

        The Compiler initializes the NcProcBuilder with the ProcModel,
        VarInitializers and PortInitializers.

        At deployment to a node, the Builder.build(..) gets executed
        resulting in the following:
          1. ProcModel gets instantiated
          2. Vars are initialized and assigned to ProcModel

        Returns
        -------
        AbstractNcProcessModel


        Raises
        ------
        NotImplementedError
        """

        pm = self.proc_model(self.proc_params)
        pm.model_id = self._model_id

        # Initialize Vars
        for name, v in self.vars.items():
            # Build variable
            lt = self._get_lava_type(name)
            if issubclass(lt.cls, np.ndarray):
                var = lt.cls(v.shape, lt.d_type)
                var[:] = v.value
            elif issubclass(lt.cls, (int, float)):
                var = v.value
            else:
                raise NotImplementedError

            # Create dynamic variable attribute on ProcModel
            setattr(pm, name, var)
            # Create private attribute for variable precision
            setattr(pm, "_" + name + "_p", lt.precision)

            pm.var_id_to_var_map[v.var_id] = name

        return pm


class RuntimeServiceBuilder(AbstractRuntimeServiceBuilder):
    """Run Time Service Builder"""

    def __init__(
        self,
        rs_class: ty.Type[AbstractRuntimeService],
        protocol: ty.Type[AbstractSyncProtocol],
        runtime_service_id: int,
        model_ids: ty.List[int],
        loihi_version: ty.Type[LoihiVersion],
        loglevel: int = logging.WARNING
    ):
        super(RuntimeServiceBuilder, self).__init__(rs_class, protocol)
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
        self._runtime_service_id = runtime_service_id
        self._model_ids: ty.List[int] = model_ids
        self.csp_send_port: ty.Dict[str, CspSendPort] = {}
        self.csp_recv_port: ty.Dict[str, CspRecvPort] = {}
        self.csp_proc_send_port: ty.Dict[str, CspSendPort] = {}
        self.csp_proc_recv_port: ty.Dict[str, CspRecvPort] = {}
        self.loihi_version: ty.Type[LoihiVersion] = loihi_version

    @property
    def runtime_service_id(self):
        return self._runtime_service_id

    def set_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Set CSP Ports

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]

        """
        for port in csp_ports:
            if isinstance(port, CspSendPort):
                self.csp_send_port.update({port.name: port})
            if isinstance(port, CspRecvPort):
                self.csp_recv_port.update({port.name: port})

    def set_csp_proc_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Set CSP Process Ports

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]

        """
        for port in csp_ports:
            if isinstance(port, CspSendPort):
                self.csp_proc_send_port.update({port.name: port})
            if isinstance(port, CspRecvPort):
                self.csp_proc_recv_port.update({port.name: port})

    def build(self,
              loihi_version: LoihiVersion = LoihiVersion.N3
              ) -> AbstractRuntimeService:
        """Build the runtime service

        Returns
        -------
        A concreate instance of AbstractRuntimeService
        [PyRuntimeService or NxSdkRuntimeService]
        """

        self.log.debug("RuntimeService Class: " + str(self.rs_class))
        nxsdk_rts = False
        if self.rs_class == NxSdkRuntimeService:
            rs = self.rs_class(protocol=self.sync_protocol,
                               loihi_version=loihi_version)
            nxsdk_rts = True
            self.log.debug("Initilized NxSdkRuntimeService")
        else:
            rs = self.rs_class(protocol=self.sync_protocol)
            self.log.debug("Initilized PyRuntimeService")
        rs.runtime_service_id = self._runtime_service_id
        rs.model_ids = self._model_ids

        if not nxsdk_rts:
            for port in self.csp_proc_send_port.values():
                if "service_to_process" in port.name:
                    rs.service_to_process.append(port)

            for port in self.csp_proc_recv_port.values():
                if "process_to_service" in port.name:
                    rs.process_to_service.append(port)

            self.log.debug("Setup 'RuntimeService <--> Rrocess; ports")

        for port in self.csp_send_port.values():
            if "service_to_runtime" in port.name:
                rs.service_to_runtime = port

        for port in self.csp_recv_port.values():
            if "runtime_to_service" in port.name:
                rs.runtime_to_service = port

        self.log.debug("Setup 'Runtime <--> RuntimeService' ports")

        return rs


@dataclass
class ChannelBuilderMp(AbstractChannelBuilder):
    """A ChannelBuilder assuming Python multi-processing is used as messaging
    and multi processing backbone.
    """
    channel_type: ChannelType
    src_process: "AbstractProcess"
    dst_process: "AbstractProcess"
    src_port_initializer: PortInitializer
    dst_port_initializer: PortInitializer

    def build(self, messaging_infrastructure: MessageInfrastructureInterface) \
            -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface

        Returns
        -------
        Channel
            PyPyChannel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type)
        return channel_class(
            messaging_infrastructure,
            self.src_port_initializer.name,
            self.dst_port_initializer.name,
            self.src_port_initializer.shape,
            self.src_port_initializer.d_type,
            self.src_port_initializer.size,
        )


@dataclass
class ServiceChannelBuilderMp(AbstractChannelBuilder):
    """A RuntimeServiceChannelBuilder assuming Python multi-processing is used
    as messaging and multi processing backbone.
    """
    channel_type: ChannelType
    src_process: ty.Union[AbstractRuntimeServiceBuilder,
                          "AbstractProcessModel"]
    dst_process: ty.Union[AbstractRuntimeServiceBuilder,
                          "AbstractProcessModel"]
    port_initializer: PortInitializer

    def build(self, messaging_infrastructure: MessageInfrastructureInterface) \
            -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface

        Returns
        -------
        Channel
            PyPyChannel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type)

        channel_name: str = (
            self.port_initializer.name
        )
        return channel_class(
            messaging_infrastructure,
            channel_name + "_src",
            channel_name + "_dst",
            self.port_initializer.shape,
            self.port_initializer.d_type,
            self.port_initializer.size,
        )


@dataclass
class RuntimeChannelBuilderMp(AbstractChannelBuilder):
    """A RuntimeChannelBuilder assuming Python multi-processing is
    used as messaging and multi processing backbone.
    """
    channel_type: ChannelType
    src_process: ty.Union[AbstractRuntimeServiceBuilder, ty.Type["Runtime"]]
    dst_process: ty.Union[AbstractRuntimeServiceBuilder, ty.Type["Runtime"]]
    port_initializer: PortInitializer

    def build(self, messaging_infrastructure: MessageInfrastructureInterface) \
            -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface

        Returns
        -------
        Channel
            PyPyChannel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type)

        channel_name: str = (
            self.port_initializer.name
        )
        return channel_class(
            messaging_infrastructure,
            channel_name + "_src",
            channel_name + "_dst",
            self.port_initializer.shape,
            self.port_initializer.d_type,
            self.port_initializer.size,
        )