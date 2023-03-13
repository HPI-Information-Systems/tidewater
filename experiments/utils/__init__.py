import pandas as pd
from typing import List, Optional, Tuple, NamedTuple, Type
from pathlib import Path
from scp import SCPClient
from paramiko import SSHClient
import tempfile

from tidewater.pipeline import Pipeline, TransformerID
from tidewater.transformers.metrics import RandScore
from tidewater.transformers.data_handling.writer import ScalarDataFrameAppender, TimeAndScalarDataFrameAppender


class RemoteLocation(NamedTuple):
    host: str
    path: Path


def collect_results(remotes: List[RemoteLocation]) -> pd.DataFrame:
    results: List[pd.DataFrame] = []
    for remote in remotes:
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(remote.host)

            transport = ssh.get_transport()
            if transport is None:
                raise ValueError("Transport is None")
            with SCPClient(transport) as scp:
                with tempfile.NamedTemporaryFile("w") as tmp:
                    scp.get(remote.path, local_path=tmp.name)
                    results.append(pd.read_csv(tmp.name))
    return pd.concat(results, ignore_index=True)


def write_result_to_file(
    results_path: Path,
    p: Pipeline,
    true_labels: Tuple[TransformerID, str],
    pred_labels: Tuple[TransformerID, str],
    meta: dict,
    print_info: str = "",
    timed: Optional[TransformerID] = None,
    metric: Optional[Type] = None,
):
    metric = metric or RandScore
    qmetric = p.add_transformer(
        metric(
            print_info=print_info,
            meta=meta,
        )
    )

    p.add_connection(true_labels[0], qmetric, (true_labels[1], "true_labels"))
    p.add_connection(pred_labels[0], qmetric, (pred_labels[1], "pred_labels"))

    if timed is not None:
        writer = p.add_transformer(TimeAndScalarDataFrameAppender(results_path))
        p.add_connection(timed, writer, ("time", "time"))
    else:
        writer = p.add_transformer(ScalarDataFrameAppender(results_path))
    p.add_connection(qmetric, writer, ("score", "data"))
