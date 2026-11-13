"""Zhong et al. (2025) data from the Figshare API, cached on disk and tabulated in DuckDB."""

from functools import lru_cache
from hashlib import md5
from pathlib import Path
import os
import warnings

import duckdb
import numpy as np
import pandas as pd
import requests

API = "https://api.figshare.com/v2/articles/28811129/versions/2"

BEHAVIOR = {
    ("supervised", "before"): "Beh_sup_train1_before_learning.npy",
    ("supervised", "after"): "Beh_sup_train1_after_learning.npy",
    ("unsupervised", "before"): "Beh_unsup_train1_before_learning.npy",
    ("unsupervised", "after"): "Beh_unsup_train1_after_learning.npy",
}

AREAS = {8: "V1", 0: "mHV", 1: "mHV", 2: "mHV", 9: "mHV", 5: "lHV", 6: "lHV", 3: "aHV", 4: "aHV"}


class Zhong:
    def __init__(self, root: str | Path = os.environ.get("ZHONG_DATA", "~/zhong2025")):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.root / "zhong.duckdb"))
        if "files" not in self.tables():
            listing = pd.DataFrame(requests.get(API, timeout=60).json()["files"])
            self.save("files", listing[["id", "name", "size", "supplied_md5", "download_url"]])
        if "sessions" not in self.tables():
            self._ingest()

    def sql(self, query: str, *params) -> pd.DataFrame:
        return self.db.execute(query, list(params)).df()

    def tables(self) -> list[str]:
        return [name for (name,) in self.db.execute("SELECT table_name FROM duckdb_tables()").fetchall()]

    def save(self, name: str, frame: pd.DataFrame) -> None:
        self.db.register("incoming", frame)
        self.db.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
        self.db.unregister("incoming")

    @property
    def files(self) -> pd.DataFrame:
        return self.sql("SELECT * FROM files ORDER BY name")

    @property
    def sessions(self) -> pd.DataFrame:
        return self.sql("SELECT * FROM sessions ORDER BY condition, mouse, phase DESC")

    def frames(self, session: str) -> pd.DataFrame:
        return self.sql("SELECT * FROM frames WHERE session = ? ORDER BY frame", session)

    def trials(self, session: str) -> pd.DataFrame:
        return self.sql("SELECT * FROM trials WHERE session = ? ORDER BY trial", session)

    def neurons(self, session: str) -> pd.DataFrame:
        return self.sql("SELECT * FROM neurons WHERE session = ? ORDER BY neuron", session)

    def fetch(self, name: str) -> Path:
        path = self.root / name
        if path.exists():
            return path
        url, checksum = self.db.execute("SELECT download_url, supplied_md5 FROM files WHERE name = ?", [name]).fetchone()
        partial = path.with_name(name + ".part")
        digest = md5()
        with requests.get(url, stream=True, timeout=120) as response, partial.open("wb") as out:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 22):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != checksum:
            partial.unlink()
            raise OSError(f"{name} does not match its Figshare checksum")
        return partial.rename(path)

    @lru_cache(maxsize=2)
    def svd(self, session: str) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings(action="ignore"):
            loaded = np.load(self.fetch(f"{session}_SVD_dec.npy"), allow_pickle=True).item()
        return loaded["U"], loaded["V"]

    def activity(self, session: str, neurons=slice(None), frames=slice(None)) -> np.ndarray:
        U, V = self.svd(session)
        return U[:, neurons].T @ V[:, frames]

    def _ingest(self) -> None:
        sessions, frames, trials, neurons = [], [], [], []
        for (condition, phase), name in BEHAVIOR.items():
            for session, b in np.load(self.fetch(name), allow_pickle=True).item().items():
                walls = dict(zip(b["stim_id"], b["UniqWalls"]))
                seconds = lambda days: (days - b["ft"][0]) * 86400
                sessions.append({
                    "session": session, "mouse": session.split("_")[0], "condition": condition, "phase": phase,
                    "leaf": walls[2], "circle": walls[0], "trials": int(b["ntrials"]), "frames": len(b["ft"]),
                })
                frames.append(pd.DataFrame({
                    "session": session,
                    "frame": np.arange(len(b["ft"])),
                    "time": seconds(b["ft"]),
                    "trial": pd.Series(b["ft_trInd"]).astype("Int64"),
                    "position": b["ft_Pos"],
                    "wall": pd.Series(b["ft_WallID"]).replace("nan", None),
                    "role": pd.Series(b["ft_WallID"]).map({walls[2]: "leaf", walls[0]: "circle"}),
                    "moving": b["ft_move"] > 0,
                    "speed": b["ft_RunSpeed"],
                }))
                trials.append(pd.DataFrame({
                    "session": session,
                    "trial": b["trInd"],
                    "wall": b["WallName"],
                    "rewarded": b["isRew"],
                    "start": seconds(b["Trial_start_time"]),
                    "end": seconds(b["Trial_end_time"]),
                    "licks": np.bincount(b["LickTrind"].astype(int), minlength=int(b["ntrials"]))[b["trInd"]],
                }))
                trans = np.load(self.fetch(f"{session.rsplit('_', 1)[0]}_trans.npz"))
                neurons.append(pd.DataFrame({
                    "session": session,
                    "neuron": np.arange(len(trans["iarea"])),
                    "area": pd.Series(trans["iarea"]).map(AREAS),
                    "x": trans["xy_t"][:, 0],
                    "y": trans["xy_t"][:, 1],
                }))
        for name, parts in [("frames", frames), ("trials", trials), ("neurons", neurons), ("sessions", [pd.DataFrame(sessions)])]:
            self.save(name, pd.concat(parts, ignore_index=True))
