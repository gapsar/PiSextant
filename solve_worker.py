#!/usr/bin/env python3
"""Solve worker -- runs on the Pi in the solver venv (cedar-solve / tetra3).

Loads a solve config and its tetra3 database once, then serves detect+solve
requests over a Unix socket (from capture_run_wave.py during live runs, later also
from the replay script). On connect it sends the client its full config, so
the run logs always record what actually ran. Responses carry the complete
cedar-detect and tetra3 outputs plus per-stage timings.

Usage:
    python solve_worker.py [configs/solver.json]
"""

import inspect
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
import tetra3
from tetra3 import cedar_detect_client

SOCKET_PATH = "/tmp/pisextant_solver.sock"
# cedar-solve ships the cedar-detect-server executable inside the tetra3
# package; set CEDAR_DETECT_BINARY to use another build.
CEDAR_BINARY = os.environ.get(
    "CEDAR_DETECT_BINARY",
    str(Path(tetra3.__file__).resolve().parent / "bin" / "cedar-detect-server"))
REPO_DIR = Path(__file__).resolve().parent
CEDAR_PORT = 50051


# --- Socket helpers (length-prefixed pickle, shared with capture_run_wave.py) ----

def send_msg(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(len(data).to_bytes(4, "big") + data)


def recv_msg(conn):
    header = _recv_exact(conn, 4)
    return pickle.loads(_recv_exact(conn, int.from_bytes(header, "big")))


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf


def sanitize(obj):
    """Convert numpy types so the client can dump the response straight to JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# --- Setup ------------------------------------------------------------------

def load_config(path):
    cfg = json.loads(Path(path).read_text())
    db = Path(cfg["database"]["path"]).expanduser()
    if not db.is_absolute():          # relative paths start at the repo root
        db = REPO_DIR / db
    if db.suffix == ".npz":   # tetra3 wants the path without the extension
        db = db.with_suffix("")
    if not db.with_suffix(".npz").exists():
        sys.exit(f"Database not found: {db.with_suffix('.npz')} -- "
                 f"fix database.path in {path}")
    return cfg, db


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_DIR / "configs" / "solver.json")
    cfg, db = load_config(config_path)
    detect_cfg = cfg["detect"]
    solve_cfg = cfg["solve"]

    print(f"Loading database {db.with_suffix('.npz')} ...")
    t3 = tetra3.Tetra3(load_database=db)
    # Filter kwargs against the installed tetra3 version's signature.
    solve_kw_names = set(inspect.signature(t3.solve_from_centroids).parameters)
    cedar = cedar_detect_client.CedarDetectClient(
        binary_path=CEDAR_BINARY, port=CEDAR_PORT)
    print(f"Solver ready: config '{cfg.get('name')}', "
          f"sigma={detect_cfg['sigma']}, binning={detect_cfg.get('binning')}")

    def solve_frame(gray):
        h, w = gray.shape
        t0 = time.perf_counter()
        centroids = cedar.extract_centroids(
            gray,
            sigma=float(detect_cfg["sigma"]),
            use_binned=(int(detect_cfg.get("binning", 1)) == 2),
            detect_hot_pixels=True,
        )
        t_detect_ms = (time.perf_counter() - t0) * 1000
        if len(centroids) == 0:
            return "no_stars", [], None, {"detect": t_detect_ms, "solve": 0.0}

        kw = {
            "size": (h, w),
            "fov_estimate": solve_cfg.get("fov_estimate"),
            "fov_max_error": solve_cfg.get("fov_max_error"),
            "pattern_checking_stars": solve_cfg.get("pattern_checking_stars", 8),
            "match_radius": solve_cfg.get("match_radius", 0.01),
            "match_threshold": solve_cfg.get("match_threshold", 1e-3),
            "solve_timeout": solve_cfg.get("solve_timeout", 3000),
            "distortion": solve_cfg.get("distortion", 0),
            "return_catalog": False,
        }
        kw = {k: v for k, v in kw.items() if k in solve_kw_names}
        max_centroids = int(detect_cfg.get("max_centroids", 30))
        t0 = time.perf_counter()
        solution = t3.solve_from_centroids(centroids[:max_centroids], **kw)
        t_solve_ms = (time.perf_counter() - t0) * 1000
        solved = solution is not None and solution.get("RA") is not None
        status = "success" if solved else "no_solution"
        return status, centroids, solution, {"detect": t_detect_ms, "solve": t_solve_ms}

    def handle_client(conn):
        send_msg(conn, {"type": "hello", "config": cfg,
                        "database": str(db.with_suffix(".npz"))})
        while True:
            request = recv_msg(conn)
            t_total0 = time.perf_counter()
            frame_id = request.get("frame_id")
            path = request.get("path")

            if not path or not os.path.exists(path):
                send_msg(conn, {"frame_id": frame_id, "status": "file_missing",
                                "n_centroids": 0, "centroids": [],
                                "solution": None, "timings": {}})
                continue

            gray = np.load(path)
            os.remove(path)
            try:
                status, centroids, solution, timings = solve_frame(gray)
            except Exception as e:
                # Transient failures (e.g. cedar-detect's hardcoded 2 s RPC
                # deadline tripping under load) cost one frame, not the run.
                print(f"frame {frame_id}: error: {e}")
                status, centroids, solution = "error", [], None
                timings = {"detect": 0.0, "solve": 0.0}
            timings["total"] = (time.perf_counter() - t_total0) * 1000

            send_msg(conn, sanitize({
                "frame_id": frame_id,
                "status": status,
                "n_centroids": len(centroids),
                "centroids": centroids,
                "solution": solution,
                "timings": timings,
            }))
            print(f"frame {frame_id}: {status}, {len(centroids)} centroids, "
                  f"detect {timings['detect']:.0f} ms, solve {timings['solve']:.0f} ms")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print(f"Listening on {SOCKET_PATH}")

    try:
        while True:
            conn, _ = server.accept()
            print("Client connected.")
            try:
                handle_client(conn)
            except (ConnectionError, EOFError):
                print("Client disconnected.")
            except Exception as e:
                print(f"Worker error: {e}")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


if __name__ == "__main__":
    main()
