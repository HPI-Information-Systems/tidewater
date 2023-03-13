from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence

import numpy as np

from .interface import InputInterface, OutputInterface, InterfaceID, InterfaceValue, InterfaceValueType
from ..datatypes.model import Model


class TransformerMode(Enum):
    """
    This Enum tells whether a Transformer is in Training or in Transforming mode.

    Options
    -------
    TRAINING
        Transformer trains
    TRANSFORMING
        Transformer transforms (for semi/supervised transformers usually after being trained)
    """

    TRAINING = 0
    TRANSFORMING = 1


class TrainingMode(Enum):
    """
    This Enum tells how a Transformer is training.

    Options
    -------
    UNSUPERVISED
        No training is performed
    SUPERVISED
        Training requires Labels
    SEMISUPERVISED
        Training requires no Labels
    """

    UNSUPERVISED = 0
    SUPERVISED = 1
    SEMISUPERVISED = 2


class Transformer(ABC):
    """
    The base class for every transforming element in a Pipeline.
    Generally, a Transformer holds two Interfaces; an InputInterface
    for the definition and storing of incoming data and an OutputInterface for the definition and
    storing of generated data for further downstream tasks.

    When creating subclasses from the Transformer class, make sure to overwrite the
    `_build_output_interface` and `_build_input_interface` methods in order to define the correct
    input and output values.

    Attributes
    ----------
    _transform_input_interface : InputInterface
        An object for defining and storing input values
    _transform_output_interface : OutputInterface
        An object for defining and storing output values
    _transformer_mode : TransformerMode
        Determining whether the transformer trains or executes
    """

    def __init__(
        self,
        _transformer_mode: TransformerMode = TransformerMode.TRANSFORMING,
        _training_mode: TrainingMode = TrainingMode.UNSUPERVISED,
    ) -> None:
        self._build_interfaces()

        self._transformer_mode: TransformerMode = _transformer_mode
        self._training_mode: TrainingMode = _training_mode

        self._temp_file: Optional[Any] = tempfile.NamedTemporaryFile()
        self._intermediate_results_dir = Path(self._temp_file.name)

    def _build_interfaces(self) -> None:
        self._transform_input_interface = self._build_transform_input_interface()
        self._transform_output_interface = self._build_transform_output_interface()

        self._train_input_interface = self._build_train_input_interface()
        self._train_output_interface = self._build_train_output_interface()

    def set_results_dir(self, _results_dir: Optional[Union[Path, os.PathLike]]) -> None:
        if _results_dir is not None:
            self._intermediate_results_dir = Path(_results_dir)
            self._temp_file = None

    def set_input_value(self, **values: Optional[InterfaceValue]) -> None:
        """
        Sets one or more incoming values.

        Parameters
        ----------
        values: Dict[InterfaceID, Optional[InterfaceValue]]
            Incoming values named by their common identifier
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            input_interface = self._transform_input_interface
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            input_interface = self._train_input_interface

        for k, v in values.items():
            input_interface.set_value(k, v)

    def set_output_value(self, **values: Optional[InterfaceValue]) -> None:
        """
        Sets one or more outgoing values.

        Parameters
        ----------
        values: Dict[InterfaceID, Optional[InterfaceValue]]
            Outgoing values named by their common identifier
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            output_interface = self._transform_output_interface
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            output_interface = self._train_output_interface

        for k, v in values.items():
            output_interface.set_value(k, v)

    def get_input_value(self, *keys: InterfaceID) -> Sequence[Optional[InterfaceValue]]:
        """
        Gets one or more input values.

        Parameters
        ----------
        keys: List[InterfaceID]
            Input value identifiers

        Returns
        -------
        List[Optional[InterfaceValue]]
            List of values for the given keys
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            input_interface = self._transform_input_interface
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            input_interface = self._train_input_interface

        return [input_interface.get_value(key) for key in keys]

    def get_output_value(self, *keys: InterfaceID) -> Sequence[Optional[InterfaceValue]]:
        """
        Gets one or more output values.

        Parameters
        ----------
        keys: List[InterfaceID]
            Output value identifiers

        Returns
        -------
        List[Optional[InterfaceValue]]
            List of values for the given keys
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            output_interface = self._transform_output_interface
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            output_interface = self._train_output_interface

        return [output_interface.get_value(key) for key in keys]

    def get_output_interface(self) -> OutputInterface:
        """
        Gets the entire OutputInterface.

        Returns
        -------
        OutputInterface
            An object for defining and storing output values
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            return self._transform_output_interface
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            return self._train_output_interface

    @property
    def is_ready_for_execution(self) -> bool:
        """
        Checks whether the Input Interface has already received all input values.

        Returns
        -------
        bool
            Giving information about the availability of all objects
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            return self._transform_input_interface.all_available
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            return self._train_input_interface.all_available

    @abstractmethod
    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        """
        Starts the training of the Transformer.

        Parameters
        ----------
        X : np.ndarray
            Input data to be trained on
        y : Optional[np.ndarray]
            Optional Labels for input data
        kwargs: Dict[str, Any]
            Arguments for setting specific training options

        Returns
        -------
        Model
            Trained model
        """

        ...

    @abstractmethod
    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Starts the transforming of the Transformer.

        Parameters
        ----------
        X : np.ndarray
            Input data to be transformed
        kwargs : Dict[str, Any]
            Arguments for setting specific training options

        Returns
        -------
        np.ndarray
            Transformed data
        """

        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> None:
        """
        Starts the execution of the Transformer
        The input interface should be entirely available (check with `self.is_ready_for_execution`).
        Reads from `self._transform_input_interface` and writes to `self._transform_output_interface`.

        Depending on the TransformerMode, this function either calls the *_train* or the *_transform* method.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Arguments for setting specific execution options
        """

        ...

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        """
        Returns the InputInterface for this Transformer.
        This class should be overwritten when creating subclasses from Transformer.

        Returns
        -------
        InputInterface
            An object for defining and storing input values
        """

        return InputInterface(data=InterfaceValue)  # type: ignore  # does not notice Union

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        """
        Returns the OutputInterface for this Transformer.
        This class should be overwritten when creating subclasses from Transformer.

        Returns
        -------
        OutputInterface
            An object for defining and storing output values
        """

        return OutputInterface(data=InterfaceValue)  # type: ignore  # does not notice Union

    @staticmethod
    def _build_train_input_interface() -> InputInterface:
        """
        Returns the InputInterface for this Transformer.
        This class should be overwritten when creating subclasses from Transformer.

        Returns
        -------
        InputInterface
            An object for defining and storing input values
        """

        return InputInterface(data=InterfaceValue, target=InterfaceValueType)  # type: ignore  # does not notice Union

    @staticmethod
    def _build_train_output_interface() -> OutputInterface:
        """
        Returns the OutputInterface for this Transformer.
        This class should be overwritten when creating subclasses from Transformer.

        Returns
        -------
        OutputInterface
            An object for defining and storing output values
        """

        return OutputInterface(model=InterfaceValue)  # type: ignore  # does not notice Union

    @classmethod
    def Training(cls, _training_mode: TrainingMode = TrainingMode.UNSUPERVISED) -> Transformer:
        return cls(_transformer_mode=TransformerMode.TRAINING, _training_mode=_training_mode)

    @classmethod
    def Transforming(cls) -> Transformer:
        return cls(_transformer_mode=TransformerMode.TRANSFORMING)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)
