from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple, Any, Dict, Union, Optional, ContextManager, Type

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time

from .distributed import Distributed
from .transformer_id import TransformerID
from ..datatypes.model import Model
from ..transformers.base import Transformer
from ..datatypes.base import NumpyType, BaseDataType
from ..transformers.interface import InterfaceID, OutputInterface
from ..transformers.helpers.timer import Timer


@dataclass
class EdgeAttributes:
    """
    This is an attribute for a connection edge. It defines the data flow between Interfaces.

    Attributes
    ----------
    output_interface : InterfaceID
        Interface ID of outgoing data
    input_interface : InterfaceID
        Interface ID of incoming data
    """

    output_interface: InterfaceID
    input_interface: InterfaceID

    def __iter__(self) -> Iterator[InterfaceID]:
        return iter([self.output_interface, self.input_interface])


@dataclass
class Pipeline:  # todo: check for cycle and unconnected subgraphs
    """
    A Pipeline holds multiple Transformers and connects their data flow in order to build a holistic data process.
    This Transformer is always in TRANSFORMING mode.

    Attributes
    ----------
    finished : bool
        Is the Pipeline finished running?
    _graph : nx.DiGraph
        Directed Graph connecting the Transformers
    _transformers : List[Transformer]
        All included Transformers
    _training_results : List[NumpyType]
        Intermediate results for the training steps
    _execution_results : List[NumpyType]
        Intermediate results for the execution steps
    results_path : Optional[Union[Path, os.PathLike]]
        Directory to store intermediate models and data
    """

    finished: bool = False
    results_path: Optional[Union[Path, os.PathLike]] = None
    verbose: bool = True
    distributed: Optional[Distributed] = None
    _graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    _transformers: List[Transformer] = field(default_factory=list)
    _training_results: Dict[TransformerID, OutputInterface] = field(default_factory=dict)
    _execution_results: Dict[TransformerID, OutputInterface] = field(default_factory=dict)
    _execution_order: List[TransformerID] = field(default_factory=list)

    def add_transformer(self, t: Transformer, name: Optional[str] = None) -> TransformerID:
        """
        Adds a Transformer to the Pipeline.

        Parameters
        ----------
        t : Transformer
            A Transformer processing data in the Pipeline

        Returns
        -------
        TransformerID
            The added Transformer's identifier
        """

        name = name or str(t)
        t.set_results_dir(self.results_path)
        self._transformers.append(t)
        t_id = TransformerID(len(self._transformers) - 1, name=name)
        self._graph.add_node(t_id)
        return t_id

    def add_timed_transformer(self, t: Transformer) -> TransformerID:
        """
        Adds a timed Transformer to the Pipeline.

        Parameters
        ----------
        t : Transformer
            A Transformer processing data in the Pipeline

        Returns
        -------
        TransformerID
            The added Transformer's identifier
        """

        return self.add_transformer(Timer(t), name=str(t))

    def get_transformer(self, transformer_id: TransformerID) -> Transformer:
        """
        Gets a Transformer by applying an identifier.

        Parameters
        ----------
        transformer_id : TransformerID
            The Transformer's identifier

        Returns
        -------
        Transformer
            The Transformer the transformer_id points to
        """

        return self._transformers[transformer_id.t_index]

    def add_connection(
        self, out_id: TransformerID, in_id: TransformerID, edge_attr: Union[EdgeAttributes, Tuple[str, str]]
    ) -> None:
        """
        Connects to added Transformers.

        Parameters
        ----------
        out_id : TransformerID
            The data-sending Transformer's identifier
        in_id : TransformerID
            The data-receiving Transformer's identifier
        edge_attr : EdgeAttributes
            The attributes for the edge telling which Interface value to send.
        """

        if not isinstance(edge_attr, EdgeAttributes):
            edge_attr = EdgeAttributes(*edge_attr)

        self._graph.add_edge(out_id, in_id, edge_attr=edge_attr)

    def plot(self) -> None:
        """
        Plots the Pipeline graph.
        """
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(self._graph, prog="dot")

        if pos is None:
            raise ValueError("Graphviz is not installed. Please install it to use this feature.")

        if type(list(pos.keys())[0]) == str:
            nodes = {str(n): n for n in self._graph.nodes()}
            pos = {nodes[k]: v for k, v in pos.items()}

        nx.draw(self._graph, pos, with_labels=True)
        plt.show()

    def _get_starting_transformers(self) -> List[TransformerID]:
        """
        Gets the Transformers that start the Pipeline and do not have incoming edges.

        Returns
        -------
        List[TransformerID]
            List of identifiers of starting transformers
        """

        return [node for node, in_degree in self._graph.in_degree if in_degree == 0]

    def _get_ending_transformers(self) -> List[TransformerID]:
        """
        Gets the Transformers that end the Pipeline and do not have outgoing edges.

        Returns
        -------
        List[TransformerID]
            List of identifiers of ending transformers
        """

        return [node for node, out_degree in self._graph.out_degree if out_degree == 0]

    def _get_previous_transformers(self, t: TransformerID) -> List[TransformerID]:
        """
        Gets the Transformers that have an outgoing edge into the transformer of the given identifier.

        Parameters
        ----------
        t : TransformerID
            Transformer identifier whose incoming connections will be returned

        Returns
        -------
        List[TransformerID]
            List of identifiers of transformers sending data to the given transformer
        """

        return [neighbor for neighbor, _ in self._graph.in_edges(t)]  # type: ignore

    def _get_next_transformers(self, t: TransformerID) -> List[TransformerID]:
        """
        Gets the Transformers that have an incoming edge from the transformer of the given identifier.

        Parameters
        ----------
        t : TransformerID
            Transformer identifier whose outgoing connections will be returned

        Returns
        -------
        List[TransformerID]
            List of identifiers of transformers receiving data from the given transformer
        """
        neighbors = {neighbor for _, neighbor in self._graph.out_edges(t)}  # type: ignore

        return list(neighbors)

    def _get_out_edges(self, t: TransformerID) -> List[Tuple[TransformerID, EdgeAttributes]]:
        """
        Gets the outgoing edges from the transformer of the given identifier.

        Parameters
        ----------
        t : TransformerID
            Transformer identifier whose outgoing edges will be returned

        Returns
        -------
        List[Tuple[TransformerID, EdgeAttributes]]
            List of outgoing edges from the given transformer
        """

        out_edges = self._graph.out_edges(t)
        attrs = nx.get_edge_attributes(self._graph, "edge_attr")
        counter: Counter = Counter(out_edges)
        return [(e[1], attrs[(*e, c)]) for e in out_edges for c in range(counter[e])]

    def _build_execution_order(self) -> List[TransformerID]:
        """
        Builds the order in which the transformers will be executed. This order makes sure, that all incoming values are
        available at each step.

        Returns
        -------
        List[TransformerID]
            List of transformer identifiers to be executed in that order
        """

        current_nodes = self._get_starting_transformers()
        order = []
        while len(current_nodes) > 0:
            t_id = current_nodes.pop(0)
            order.append(t_id)
            for nxt in self._get_next_transformers(t_id):
                prevs = self._get_previous_transformers(nxt)
                if len(set(prevs) - set(order)) == 0:
                    current_nodes = [nxt] + current_nodes
        self._execution_order = order

        if self.distributed is not None:
            self.distributed.add_execution_order(self._execution_order)

        return self._execution_order

    def _validate_graph(self) -> None:
        """
        This method validates whether the pipeline graph has cycles or non-connected subgraphs.
        """
        try:
            nx.find_cycle(self._graph)
            raise ValueError("The pipeline has a cycle. This is not valid.")
        except nx.NetworkXNoCycle:
            pass

        subgraphs = list(nx.connected_components(self._graph.to_undirected(as_view=True)))
        if len(subgraphs) > 1:
            raise ValueError("The pipeline has disconnected components. This is not valid.")

    def _transformer_is_ready_for_execution(self, t_id: TransformerID) -> bool:
        transformer = self.get_transformer(t_id)

        if self.distributed is not None:
            while not transformer.is_ready_for_execution:
                input_values = self.distributed.get_input_values(t_id)
                transformer.set_input_value(**input_values)
                time.sleep(1)
            return True
        return transformer.is_ready_for_execution

    def _set_input_values_for_transformer(
        self, t_id: TransformerID, **input_values: Union[List[BaseDataType], BaseDataType]
    ) -> None:
        if self.distributed is not None:
            self.distributed.set_input_values(t_id, **input_values)
        else:
            transformer = self.get_transformer(t_id)
            transformer.set_input_value(**input_values)

    def iter_jobs(self) -> Iterator[TransformerID]:
        if self.distributed is not None:
            return self.distributed.iter_jobs()
        return iter(self._execution_order)

    def execute(self, **kwargs: Any) -> None:
        with ContextManagerWrapper(self.distributed):
            self._validate_graph()
            self._build_execution_order()
            print("starting pipeline execution")
            for i, t_id in enumerate(self.iter_jobs()):
                transformer = self.get_transformer(t_id)
                assert self._transformer_is_ready_for_execution(t_id), "Transformer is not yet ready for execution"
                print(f"[{i+1}/{len(self._execution_order)}] Execution of transformer {transformer}")
                transformer.execute()
                self._execution_results[t_id] = transformer.get_output_interface()
                if self.verbose:
                    print(self._execution_results[t_id])
                for nxt, attr in self._get_out_edges(t_id):
                    out_value = self._execution_results[t_id].get_value(attr.output_interface)
                    self._set_input_values_for_transformer(nxt, **{attr.input_interface: out_value})  # type: ignore
            self.finished = True


class ContextManagerWrapper(ContextManager):
    def __init__(self, wrapped: Optional[Any]) -> None:
        self.wrapped = wrapped

    def __enter__(self) -> Any:
        if self.wrapped is not None:
            return self.wrapped.__enter__()
        return None

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        exception_traceback: Optional[Any],
    ) -> Any:
        if self.wrapped is not None:
            return self.wrapped.__exit__(exception_type, exception_value, exception_traceback)
        return None


class Timing(ContextManager):
    def __init__(self, timeout: Optional[int] = None) -> None:
        self.timeout = timeout
        self.start_time: Optional[dt.datetime] = None
        self.duration: Optional[dt.timedelta] = None

    def __enter__(self) -> Timing:
        self.start_time = dt.datetime.now()
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        exception_traceback: Optional[Any],
    ) -> None:
        if self.start_time is not None:
            self.duration = dt.datetime.now() - self.start_time
