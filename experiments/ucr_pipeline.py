import sys
import argparse
import tempfile
from pathlib import Path
from typing import List, Optional

from tidewater.pipeline import Pipeline, Distributed
from tidewater.pipeline.base import EdgeAttributes
from tidewater.pipeline.transformer_id import TransformerID
from tidewater.transformers.clusterings.distance_metrics import DistanceMetric, Interpolation
from tidewater.transformers.clusterings.happie_clust import HappieClust
from tidewater.transformers.metrics import RandScore
from tidewater.transformers.data_handling.label_reshaper import LabelReshaper
from tidewater.transformers.clusterings import KMeans, MeanShift, Hierarchical, KMedoids, Optics, DBScan, KShape
from tidewater.transformers.clusterings.birch import Birch
from tidewater.transformers.clusterings.estimation import PreClustering, PreClusteringNoEncoding
from tidewater.transformers.clusterings.estimation.ward_dm import WardDM
from tidewater.transformers.clusterings.base import DynSizeClustering
from tidewater.transformers.encoders.base import Encoder
from tidewater.transformers.encoders.distance_space import DistanceSpace
from tidewater.transformers.encoders.shapelet_space import ShapeletSpace
from tidewater.transformers.encoders.grid_space import GridSpace
from tidewater.transformers.encoders.random_space import RandomSpace
from tidewater.transformers.encoders.median_space import MedianSpace
from tidewater.transformers.encoders.birch_space_constructor import BirchSpaceConstructor
from tidewater.transformers.clusterings.feature_based import FeatureBased
from tidewater.transformers.clusterings.centroid_computation.dba import DBA
from tidewater.transformers.data_handling import HDF5TimeSeriesLoader
from tidewater.transformers.data_handling.loader import UCRTimeSeriesLoader
from tidewater.transformers.data_handling.normalizer import Normalizer
from tidewater.transformers.data_handling.plotter import Plotter, SpacePlotter
from tidewater.transformers.data_handling.range import Range
from tidewater.transformers.data_handling.scaler import MinMaxScaler
from tidewater.transformers.data_handling.sliding_window import SlidingWindow
from tidewater.transformers.data_handling.subsequencer import Subsequencer
from tidewater.transformers.data_handling.subsequence_merger import SubsequenceMerger
from tidewater.transformers.data_handling.interpolater import Interpolater
from tidewater.transformers.data_handling.writer import TimeAndScalarDataFrameAppender, ScalarDataFrameAppender
from tidewater.transformers.clusterings.jet import JET

from .utils import write_result_to_file


def run_pipeline(
    path: Path,
    results_path: Path,
    n_clusters: int,
    n_jobs=8,
    distributed=False,
    scheduler=False,
    algorithms: Optional[List[str]] = None,
    host: str = "localhost",
    port: int = 6379,
):
    #############
    ### SETUP ###
    #############

    p = Pipeline(
        results_path=Path("./results"),
        verbose=False,
        distributed=Distributed(host=host, port=port, scheduler=scheduler) if distributed else None,
    )

    ###############
    ### LOADING ###
    ###############

    loader = p.add_transformer(UCRTimeSeriesLoader(path=path))

    ###############
    ### JET SBD ###
    ###############

    if algorithms is None or "JET-SBD" in algorithms:
        clustering = p.add_timed_transformer(
            JET(n_clusters=n_clusters, n_jobs=n_jobs, verbose=True, metric=DistanceMetric.SHAPE_BASED_DISTANCE)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "JET SBD"},
            timed=clustering,
        )

    ###############
    ### JET DTW ###
    ###############

    if algorithms is None or "JET-DTW" in algorithms:
        clustering = p.add_timed_transformer(
            JET(n_clusters=n_clusters, n_jobs=n_jobs, verbose=True, metric=DistanceMetric.DTW)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "JET DTW"},
            timed=clustering,
        )

    ###############
    ### JET MSM ###
    ###############

    if algorithms is None or "JET-MSM" in algorithms:
        clustering = p.add_timed_transformer(
            JET(n_clusters=n_clusters, n_jobs=n_jobs, verbose=True, metric=DistanceMetric.MSM, c=700)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "JET MSM"},
            timed=clustering,
        )

    ###################
    ### K-Means DTW ###
    ###################
    if algorithms is None or "K-Means-DTW" in algorithms:
        clustering = p.add_timed_transformer(KMeans(n_clusters=n_clusters, n_jobs=n_jobs))
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "K-Means DTW"},
            timed=clustering,
        )

    ###############
    ### K-Shape ###
    ###############

    if algorithms is None or "K-Shape" in algorithms:
        clustering = p.add_timed_transformer(KShape(n_clusters=n_clusters, interpolation=Interpolation.DOWN))
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "K-Shape"},
            timed=clustering,
        )

    #########################
    ### K-Means Euclidean ###
    #########################

    if algorithms is None or "K-Means-Euclidean" in algorithms:
        clustering = p.add_timed_transformer(KMeans(n_clusters=n_clusters, n_jobs=n_jobs, metric="euclidean"))
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "K-Means Euclidean"},
            timed=clustering,
        )

    #################
    ### MeanShift ###
    #################

    if algorithms is None or "MeanShift" in algorithms:
        clustering = p.add_timed_transformer(
            MeanShift(n_threads=n_jobs, distance_measure="euclidean", interpolation=Interpolation.DOWN)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "MeanShift"},
            timed=clustering,
        )

    ##############
    ### DBSCAN ###
    ##############

    if algorithms is None or "DBSCAN" in algorithms:
        clustering = p.add_timed_transformer(DBScan(n_jobs=n_jobs, interpolation=Interpolation.DOWN))
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "DBSCAN"},
            timed=clustering,
        )

    ##############
    ### OPTICS ###
    ##############

    if algorithms is None or "OPTICS" in algorithms:
        clustering = p.add_timed_transformer(Optics(n_jobs=n_jobs, interpolation=Interpolation.DOWN))
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "OPTICS"},
            timed=clustering,
        )

    #############
    ### Birch ###
    #############

    if algorithms is None or "Birch" in algorithms:
        clustering = p.add_timed_transformer(
            Birch(n_clusters=n_clusters, n_jobs=n_jobs, interpolation=Interpolation.DOWN)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "Birch"},
            timed=clustering,
        )

    #####################
    ### K-Medoids DTW ###
    #####################

    if algorithms is None or "K-Medoids-DTW" in algorithms:
        clustering = p.add_timed_transformer(KMedoids(n_clusters=n_clusters, n_jobs=n_jobs, metric=DistanceMetric.DTW))
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "KMedoids DTW"},
            timed=clustering,
        )

    #####################
    ### K-Medoids MSM ###
    #####################

    if algorithms is None or "K-Medoids-MSM" in algorithms:
        clustering = p.add_timed_transformer(
            KMedoids(n_clusters=n_clusters, n_jobs=n_jobs, metric=DistanceMetric.MSM, algorithm_args={"c": 700})
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "KMedoids MSM"},
            timed=clustering,
        )

    #####################
    ### K-Medoids SBD ###
    #####################

    if algorithms is None or "K-Medoids-SBD" in algorithms:
        clustering = p.add_timed_transformer(
            KMedoids(n_clusters=n_clusters, n_jobs=n_jobs, metric=DistanceMetric.SHAPE_BASED_DISTANCE)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "KMedoids SBD"},
            timed=clustering,
        )

    ###################################
    ### Hierarchical Clustering DTW ###
    ###################################

    if algorithms is None or "Hierarchical-DTW" in algorithms:
        clustering = p.add_timed_transformer(
            Hierarchical(n_clusters=n_clusters, n_jobs=n_jobs, verbose=True, metric=DistanceMetric.DTW, method="ward")
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "Hierarchical Clustering DTW"},
            timed=clustering,
        )

    ###################################
    ### Hierarchical Clustering MSM ###
    ###################################

    if algorithms is None or "Hierarchical-MSM" in algorithms:
        clustering = p.add_timed_transformer(
            Hierarchical(
                n_clusters=n_clusters,
                n_jobs=n_jobs,
                verbose=True,
                metric=DistanceMetric.MSM,
                method="ward",
                algorithm_args={"c": 700},
            )
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "Hierarchical Clustering MSM"},
            timed=clustering,
        )

    ###################################
    ### Hierarchical Clustering SBD ###
    ###################################

    if algorithms is None or "Hierarchical-SBD" in algorithms:
        clustering = p.add_timed_transformer(
            Hierarchical(
                n_clusters=n_clusters,
                n_jobs=n_jobs,
                verbose=True,
                metric=DistanceMetric.SHAPE_BASED_DISTANCE,
                method="ward",
            )
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "Hierarchical Clustering SBD"},
            timed=clustering,
        )

    ###################
    ### HappieClust ###
    ###################

    if algorithms is None or "HappieClust" in algorithms:
        clustering = p.add_timed_transformer(
            HappieClust(metric=DistanceMetric.SHAPE_BASED_DISTANCE, n_clusters=n_clusters, n_jobs=n_jobs, verbose=True)
        )
        p.add_connection(loader, clustering, ("data", "data"))

        write_result_to_file(
            results_path,
            p,
            (loader, "labels"),
            (clustering, "data"),
            meta={"dataset": path, "algorithm": "HappieClust"},
            timed=clustering,
        )


def generate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCR Clustering Pipeline")

    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--distributed", action="store_true", dest="distributed")
    parser.add_argument("--scheduler", action="store_true", dest="scheduler")
    # list argument for only running specific algorithms
    parser.add_argument("--algorithms", nargs="+", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.set_defaults(scheduler=False, distributed=False)

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = generate_args()
    run_pipeline(
        args.path,
        args.results_path,
        args.n_clusters,
        args.n_jobs,
        args.distributed,
        args.scheduler,
        args.algorithms,
        host=args.host,
        port=args.port,
    )
