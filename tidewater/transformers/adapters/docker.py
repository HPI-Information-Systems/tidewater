from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, Any, Optional, Union
from pathlib import Path
import tempfile

import docker
import numpy as np
from docker.models.containers import Container
from timeeval.adapters import DockerAdapter as TEDockerAdapter
from timeeval.adapters.docker import (
    AlgorithmInterface,
    DATASET_TARGET_PATH,
    RESULTS_TARGET_PATH,
    SCORES_FILE_NAME,
    MODEL_FILE_NAME,
)
from timeeval.data_types import ExecutionType
from timeeval.resource_constraints import GB

from ..base import Transformer, TransformerMode, TrainingMode
from ..interface import InputInterface, OutputInterface
from ...datatypes.base import NumpyType
from ...datatypes.model import Model


class DockerAdapter(TEDockerAdapter, Transformer, ABC):
    """
    A base Transformer for all dockerized algorithms. It uses the same initialization parameters as the
    `TimeEval`_.

     .. _TimeEval: https://github.com/HPI-Information-Systems/TimeEval/blob/main/timeeval/adapters/docker.py
    """

    def __init__(
        self,
        image_name: str,
        _transformer_mode: TransformerMode = TransformerMode.TRANSFORMING,
        _training_mode: TrainingMode = TrainingMode.UNSUPERVISED,
        **kwargs: Any,
    ) -> None:
        super().__init__(image_name, **kwargs)
        self._transform_input_interface: InputInterface = self._build_transform_input_interface()
        self._transform_output_interface: OutputInterface = self._build_transform_output_interface()

        self._train_input_interface: InputInterface = self._build_train_input_interface()
        self._train_output_interface: OutputInterface = self._build_train_output_interface()

        self._transformer_mode: TransformerMode = _transformer_mode
        self._training_mode: TrainingMode = _training_mode

    def _get_default_model_path(self) -> Path:
        return (self._intermediate_results_dir / MODEL_FILE_NAME).absolute()

    def _run_container(self, dataset_path: Path, args: dict) -> Container:
        client = docker.from_env()

        algorithm_interface = AlgorithmInterface(
            dataInput=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            dataOutput=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute(),
            modelInput=args.get("modelPath", (Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute()),
            modelOutput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
            executionType=args.get("executionType", ExecutionType.EXECUTE.value),
            customParameters=args.get("hyper_params", {}),
        )

        uid = DockerAdapter._get_uid()
        gid = DockerAdapter._get_gid(self.group)
        if not gid:
            gid = uid
        print(
            f"Running container '{self.image_name}:{self.tag}' with uid={uid} and gid={gid} privileges in {algorithm_interface.executionType} mode."
        )

        memory_limit, cpu_limit = self._get_compute_limits(args)
        cpu_shares = int(cpu_limit * 1e9)
        print(f"Restricting container to {cpu_limit} CPUs and {memory_limit / GB:.3f} GB RAM")

        args["results_path"] = self._intermediate_results_dir
        return client.containers.run(
            f"{self.image_name}:{self.tag}",
            f"execute-algorithm '{algorithm_interface.to_json_string()}'",
            volumes={
                str(dataset_path.parent.absolute()): {"bind": DATASET_TARGET_PATH, "mode": "ro"},
                str(self._results_path(args, absolute=True)): {"bind": RESULTS_TARGET_PATH, "mode": "rw"},
            },
            environment={"LOCAL_GID": gid, "LOCAL_UID": uid},
            mem_swappiness=0,
            mem_limit=memory_limit,
            memswap_limit=memory_limit,
            nano_cpus=cpu_shares,
            detach=True,
        )

    def _result_to_output(self, result: Union[Path, np.ndarray]) -> None:
        """
        Sets the output from the result parameter.

        Parameters
        ----------
        result : Union[Path, np.ndarray]
            The result to be set to output
        """

        if isinstance(result, Path):
            self.set_output_value(data=self._file_to_numpy_return_type(result))
        else:
            self.set_output_value(data=self._numpy_return_type(ndarray=result))

    def _numpy_type_to_file(self, data: NumpyType) -> Path:
        """
        Writes a NumpyType object to a temporary file and returns its path.

        Parameters
        ----------
        data : NumpyType
            Object to be written to temporary file

        Returns
        -------
        Path
            Path to temporary file holding data object
        """

        path = Path(tempfile.NamedTemporaryFile().name)
        data.to_file(path)
        return path

    def _file_to_numpy_return_type(self, path: Path) -> NumpyType:
        """
        Reads a NumpyType object from a temporary file and returns it.

        Parameters
        ----------
        path : Path
            Path to temporary file holding data object

        Returns
        -------
        NumpyType
            Object read from temporary file
        """

        return self._numpy_return_type.from_file(path)

    def _file_to_numpy(self, path: Path) -> np.ndarray:
        data: np.ndarray = np.loadtxt(str(path), delimiter=",")
        return data

    def _to_correct_numpy_format(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        assert X.ndim == 2 and (y is None or y.ndim == 2), "X and y are not in the best shape. Must be 2d."
        if y is not None:
            X = np.concatenate([X, y], axis=1)
        else:
            X = np.concatenate([X, np.zeros(X.shape[0]).reshape(-1, 1)], axis=1)
        print(X)
        data: np.ndarray = np.concatenate([np.arange(X.shape[0]).reshape(-1, 1), X], axis=1)
        return data

    @property
    @abstractmethod
    def _numpy_return_type(self) -> Type[NumpyType]:
        """
        Defines the type that the docker container returns after execution.

        Returns
        -------
        Type[NumpyType]
            Execution return type
        """

        ...

    @property
    @abstractmethod
    def _numpy_input_type(self) -> Type[NumpyType]:
        """
        Defines the type that the docker container expects for execution.

        Returns
        -------
        Type[NumpyType]
            Expected input type
        """

        ...

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType)

    @staticmethod
    def _build_transform_output_interface() -> OutputInterface:
        return OutputInterface(data=NumpyType)

    @staticmethod
    def _build_train_output_interface() -> OutputInterface:
        return OutputInterface(model=Model)


class UnsupervisedDockerAdapter(DockerAdapter):
    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        pass

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        data = self._to_correct_numpy_format(X)
        labels: Union[np.ndarray, Path] = self._call(self._numpy_type_to_file(NumpyType(ndarray=data)), kwargs)
        if isinstance(labels, np.ndarray):
            return labels
        return self._file_to_numpy(labels)

    def execute(self, **kwargs: Any) -> None:
        data = self.get_input_value("data")[0]
        assert data is not None and isinstance(data, NumpyType), "Input data is not a filled NumpyType"

        if self._transformer_mode == TransformerMode.TRAINING:
            raise ValueError("An unsupervised algorithm does not have a training phase.")
        else:
            labels = self._transform(data.to_2d())
            self._result_to_output(labels)


class SemiSupervisedDockerAdapter(DockerAdapter):
    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        data = self._to_correct_numpy_format(X)
        kwargs["executionType"] = ExecutionType.TRAIN
        _result = self._call(self._numpy_type_to_file(NumpyType(ndarray=data)), kwargs)
        return Model(path=self._get_default_model_path())

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        model = self.get_input_value("model")[0]
        assert model is not None and isinstance(model, Model), "Input model is not a filled Model"
        kwargs["modelPath"] = model.path

        data = self._to_correct_numpy_format(X)
        labels: Union[np.ndarray, Path] = self._call(self._numpy_type_to_file(NumpyType(ndarray=data)), kwargs)
        if isinstance(labels, np.ndarray):
            return labels
        return self._file_to_numpy(labels)

    @staticmethod
    def _build_train_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType, model=Model)

    def execute(self, **kwargs: Any) -> None:
        data = self.get_input_value("data")[0]
        assert data is not None and isinstance(data, NumpyType), "Input data is not a filled NumpyType"

        if self._transformer_mode == TransformerMode.TRAINING:
            model = self._train(data.to_2d(), **kwargs)
            self.set_output_value(model=model)
        else:
            labels = self._transform(data.to_2d())
            self._result_to_output(labels)


class SupervisedDockerAdapter(DockerAdapter):
    def _train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Model:
        data = self._to_correct_numpy_format(X, y)
        kwargs["executionType"] = ExecutionType.TRAIN
        _result = self._call(self._numpy_type_to_file(NumpyType(ndarray=data)), kwargs)
        return Model(path=self._get_default_model_path())

    def _transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        model = self.get_input_value("model")[0]
        assert model is not None and isinstance(model, Model), "Input model is not a filled Model"
        kwargs["modelPath"] = model.path

        data = self._to_correct_numpy_format(X)
        labels: Union[np.ndarray, Path] = self._call(self._numpy_type_to_file(NumpyType(ndarray=data)), kwargs)
        if isinstance(labels, np.ndarray):
            return labels
        return self._file_to_numpy(labels)

    @staticmethod
    def _build_train_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType, labels=NumpyType)

    @staticmethod
    def _build_transform_input_interface() -> InputInterface:
        return InputInterface(data=NumpyType, model=Model)

    def execute(self, **kwargs: Any) -> None:
        data = self.get_input_value("data")[0]
        assert data is not None and isinstance(data, NumpyType), "Input data is not a filled NumpyType"

        if self._transformer_mode == TransformerMode.TRAINING:
            labels = self.get_input_value("labels")[0]
            assert labels is not None and isinstance(labels, NumpyType), "Input labels is not a filled NumpyType"

            model = self._train(data.to_2d(), labels.to_2d(), **kwargs)
            self.set_output_value(model=model)
        else:
            predicted_labels: np.ndarray = self._transform(data.ndarray)
            self._result_to_output(predicted_labels)
