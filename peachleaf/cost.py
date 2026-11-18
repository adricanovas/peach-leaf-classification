"""Computational cost of every architecture.

Measured in float32 with batch 1 (single-image latency, the relevant figure for
in-field use) and batch 32 (throughput).  Run once per device type::

    python -m peachleaf.run cost --backbone resnet50 --variant cbam --device gpu
    python -m peachleaf.run cost --backbone resnet50 --variant cbam --device cpu --threads 4
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import socket
import statistics
import tempfile
import time

import numpy as np

from . import config as C


def count_flops(model, img_size: int = C.IMG_SIZE) -> int:
    """FLOPs of one forward pass (batch 1); one multiply-add counts as 2 FLOPs."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import \
        convert_variables_to_constants_v2_as_graph

    fn = tf.function(lambda x: model(x, training=False))
    cf = fn.get_concrete_function(tf.TensorSpec([1, img_size, img_size, 3], tf.float32))
    _, graph_def = convert_variables_to_constants_v2_as_graph(cf)
    with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        with tf.Graph().as_default() as g:
            tf.compat.v1.import_graph_def(graph_def, name="")
            prof = tf.compat.v1.profiler.profile(
                graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd="op",
                options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    return int(prof.total_float_ops)


def measure_latency(model, batch_size: int, img_size: int = C.IMG_SIZE, warmup: int = 20,
                    runs: int = 100) -> dict:
    import tensorflow as tf

    x = tf.constant(np.random.default_rng(0).uniform(0, 255, (batch_size, img_size, img_size, 3)), tf.float32)
    infer = tf.function(lambda inp: model(inp, training=False))
    for _ in range(warmup):
        tf.reduce_sum(infer(x)).numpy()
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tf.reduce_sum(infer(x)).numpy()  # forces device synchronisation
        lat.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(lat)
    return {"latency_ms_mean": statistics.mean(lat), "latency_ms_median": med,
            "latency_ms_p95": float(np.percentile(lat, 95)), "latency_ms_std": statistics.stdev(lat),
            "throughput_img_s": batch_size * 1e3 / med}


def peak_gpu_memory_mb(model, batch_size: int, img_size: int = C.IMG_SIZE):
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        return None
    tf.config.experimental.reset_memory_stats("GPU:0")
    model(tf.zeros([batch_size, img_size, img_size, 3]), training=False)
    return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20


def model_file_size_mb(model) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.keras")
        model.save(p)
        return os.path.getsize(p) / 2 ** 20


def run_cost(a):
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    from .models import build_model, count_params

    tag = f"{a.device}{a.threads or ''}"
    out = C.RUNS_ROOT / "cost" / tag / f"{a.backbone}__{a.variant}.json"
    if out.exists() and not a.force:
        return print("already done:", out)
    model = build_model(a.backbone, a.variant, weights=None)
    total, attn = count_params(model)
    row = {"backbone": a.backbone, "variant": a.variant, "device": a.device, "threads": a.threads,
           "host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
           "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
           "gpu_name": (tf.config.experimental.get_device_details(
               tf.config.list_physical_devices("GPU")[0]).get("device_name")
               if tf.config.list_physical_devices("GPU") else None),
           "params": total, "attention_params": attn, "flops": count_flops(model),
           "file_mb": model_file_size_mb(model)}
    for bs in (1, 32):
        lat = measure_latency(model, bs, runs=200 if bs == 1 else 50)
        row.update({f"bs{bs}_{k}": v for k, v in lat.items()})
        row[f"bs{bs}_peak_gpu_mb"] = peak_gpu_memory_mb(model, bs)
    row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    from .run import mkdirs
    mkdirs(out.parent)
    out.write_text(json.dumps(row, indent=1, default=float))
    print(json.dumps(row, indent=1, default=float))
