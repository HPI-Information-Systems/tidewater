from typing import Optional, Any, ContextManager, List

import time
import numpy as np

from ...datatypes.model import Model
from ...datatypes import TimeSeries, Labels, Scalar
from ..base import Transformer, InputInterface, OutputInterface, InterfaceValue, TransformerMode


class TimeContextManager(ContextManager):
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()

    @property
    def elapsed(self):
        return self.end - self.start


class Timer(Transformer):
    def __init__(self, transformer: Transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = transformer

    def execute(self, **kwargs: Any) -> None:
        with TimeContextManager() as timer:
            X = self._transformer.execute(**kwargs)
        print(f"Elapsed time: {timer.elapsed}")
        self.set_output_value(time=Scalar(data=timer.elapsed))
        return X

    def _train(self, X: TimeSeries, y: Optional[Labels] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def set_input_value(self, **values: Optional[InterfaceValue]) -> None:
        for k, v in values.items():
            if k in self._transform_input_interface or k in self._train_input_interface:
                super().set_input_value(**{k: v})
            else:
                self._transformer.set_input_value(**{k: v})

    def get_output_value(self, key: str) -> Optional[InterfaceValue]:
        if key in self._transform_output_interface or key in self._train_output_interface:
            return super().get_output_value(key)
        else:
            return self._transformer.get_output_value(key)

    def get_output_interface(self) -> OutputInterface:
        """
        Gets the entire OutputInterface.

        Returns
        -------
        OutputInterface
            An object for defining and storing output values
        """

        if self._transformer_mode == TransformerMode.TRANSFORMING:
            return self._transform_output_interface.merge(self._transformer.get_output_interface())
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            return self._train_output_interface.merge(self._transformer.get_output_interface())

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
            return self._transform_input_interface.all_available and self._transformer.is_ready_for_execution
        else:  # if self._transformer_mode == TransformerMode.TRAINING
            return self._train_input_interface.all_available and self._transformer.is_ready_for_execution

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface()

    @staticmethod
    def _build_transform_output_interface(**inherited: InterfaceValue) -> OutputInterface:
        return OutputInterface(time=Scalar)

    def __str__(self) -> str:
        return f"Timer({self._transformer})"
