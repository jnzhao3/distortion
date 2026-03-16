from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import utils_3 as ut

try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_DATASET_NAME = "lmarena-ai/arena-human-preference-140k"
DEFAULT_SPLIT = "train"
DEFAULT_CACHE_DIR = Path(".cache") / "simulation_12"
DEFAULT_DS_KEYS = [
    ("is_code",),
    ("category_tag", "creative_writing_v0.1", "creative_writing"),
    ("category_tag", "criteria_v0.1", "complexity"),
    ("category_tag", "criteria_v0.1", "creativity"),
    ("category_tag", "criteria_v0.1", "domain_knowledge"),
    ("category_tag", "criteria_v0.1", "problem_solving"),
    ("category_tag", "criteria_v0.1", "real_world"),
    ("category_tag", "criteria_v0.1", "specificity"),
    ("category_tag", "criteria_v0.1", "technical_accuracy"),
    ("category_tag", "math_v0.1", "math"),
    ("category_tag", "if_v0.1", "if"),
]
DEFAULT_KEY_NAMES = [
    "is_code",
    "creative_writing",
    "complexity",
    "creativity",
    "domain_knowledge",
    "problem_solving",
    "real_world",
    "specificity",
    "technical_accuracy",
    "math",
    "if",
]
KEY_NAME_TO_PATH = dict(zip(DEFAULT_KEY_NAMES, DEFAULT_DS_KEYS))


def bit_strings(n: int) -> np.ndarray:
    x = np.arange(2**n, dtype=np.uint32)[:, None]
    return (x >> np.arange(n - 1, -1, -1)) & 1


def resolve_ds_keys(key_names: list[str] | None) -> list[tuple[str, ...]]:
    if key_names is None:
        return list(DEFAULT_DS_KEYS)

    unknown = [name for name in key_names if name not in KEY_NAME_TO_PATH]
    if unknown:
        raise ValueError(
            f"Unknown keys: {unknown}. Available keys: {sorted(KEY_NAME_TO_PATH)}"
        )

    return [KEY_NAME_TO_PATH[name] for name in key_names]


def format_vector_key(key_vector: np.ndarray) -> str:
    return "".join(map(str, np.asarray(key_vector, dtype=np.uint8).tolist()))


@dataclass
class SubgroupResult:
    key_vector: np.ndarray
    winners: np.ndarray
    losers: np.ndarray
    r_hat: np.ndarray | None
    misspecification_error: float | None


@dataclass
class SimulationResult:
    pop_utilities: np.ndarray
    absolute_subsection_sizes: np.ndarray
    misspec_errors: dict[int, float]
    included_subgroup_indices: np.ndarray
    avg_utils: np.ndarray
    true_ranking: np.ndarray
    borda_scores: np.ndarray
    ranking: np.ndarray
    distortion: float
    k: int


class DistortionSimulation:
    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        ds_keys: list[tuple[str, ...]] | None = None,
        beta: float = 1.0,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        weight_voter_dist_by_subgroup_size: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.ds_keys = ds_keys or list(DEFAULT_DS_KEYS)
        self.beta = beta
        self.cache_dir = Path(cache_dir)
        self.weight_voter_dist_by_subgroup_size = weight_voter_dist_by_subgroup_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.train_data = None
        self.all_candidate_idxs: dict[str, int] | None = None
        self.vector_keys = bit_strings(len(self.ds_keys))

        # Cached/precomputed arrays populated by load()
        self._cache_key_value = self._cache_key()
        self._num_rows: int | None = None
        self._language_arr: np.ndarray | None = None
        self._winner_arr: np.ndarray | None = None
        self._winner_code: np.ndarray | None = None
        self._model_a_idx: np.ndarray | None = None
        self._model_b_idx: np.ndarray | None = None
        self._key_arrays: list[np.ndarray] | None = None
        self._subgroup_codes: np.ndarray | None = None

    def load(self) -> None:
        dataset = load_dataset(self.dataset_name)
        self.train_data = dataset[self.split]

        # Preserve the original candidate indexing behavior as closely as possible.
        all_candidates = set(self.train_data["model_a"]).union(set(self.train_data["model_b"]))
        self.all_candidate_idxs = {
            candidate: i for i, candidate in enumerate(all_candidates)
        }

        model_a = np.asarray(self.train_data["model_a"])
        model_b = np.asarray(self.train_data["model_b"])
        winner = np.asarray(self.train_data["winner"])
        language = np.asarray(self.train_data["language"])

        self._num_rows = len(self.train_data)
        self._language_arr = language
        self._winner_arr = winner

        self._model_a_idx = np.fromiter(
            (self.all_candidate_idxs[candidate] for candidate in model_a),
            dtype=np.int64,
            count=len(model_a),
        )
        self._model_b_idx = np.fromiter(
            (self.all_candidate_idxs[candidate] for candidate in model_b),
            dtype=np.int64,
            count=len(model_b),
        )

        winner_code = np.zeros(len(winner), dtype=np.int8)
        winner_code[winner == "model_a"] = 1
        winner_code[winner == "model_b"] = 2
        self._winner_code = winner_code

        key_arrays: list[np.ndarray] = []
        for path in self.ds_keys:
            values = self.train_data
            for key in path:
                values = values[key]
            key_arrays.append(np.asarray(values, dtype=np.uint8))
        self._key_arrays = key_arrays

        subgroup_codes = np.zeros(len(self.train_data), dtype=np.uint32)
        for arr in key_arrays:
            subgroup_codes = (subgroup_codes << 1) | arr.astype(np.uint32)
        self._subgroup_codes = subgroup_codes

    @property
    def n_items(self) -> int:
        if self.all_candidate_idxs is None:
            raise RuntimeError("Simulation data has not been loaded.")
        return len(self.all_candidate_idxs)

    def _cache_key(self) -> str:
        payload = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "ds_keys": self.ds_keys,
            "beta": self.beta,
            "weight_voter_dist_by_subgroup_size": self.weight_voter_dist_by_subgroup_size,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def config_dict(self) -> dict[str, object]:
        key_names = []
        for path in self.ds_keys:
            for name, candidate_path in KEY_NAME_TO_PATH.items():
                if candidate_path == path:
                    key_names.append(name)
                    break
            else:
                key_names.append(">".join(path))

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "ds_keys": [list(path) for path in self.ds_keys],
            "key_names": key_names,
            "beta": self.beta,
            "cache_dir": str(self.cache_dir),
            "weight_voter_dist_by_subgroup_size": self.weight_voter_dist_by_subgroup_size,
            "cache_key": self._cache_key_value,
            "num_keys": len(self.ds_keys),
            "num_subgroups": len(self.vector_keys),
        }

    def _subgroup_cache_path(self, key_vector: np.ndarray, language: str | None) -> Path:
        language_token = "all" if language is None else language
        vector_token = format_vector_key(key_vector)
        filename = f"subgroup_{self._cache_key_value}_{language_token}_{vector_token}.npz"
        return self.cache_dir / filename

    def _population_cache_path(self) -> Path:
        return self.cache_dir / f"population_{self._cache_key_value}.npz"

    def _vector_to_code(self, key_vector: np.ndarray) -> int:
        code = 0
        for bit in np.asarray(key_vector, dtype=np.uint8):
            code = (code << 1) | int(bit)
        return code

    def filter_fast(
        self,
        key_vector: np.ndarray,
        language: str | None = "en",
        show_progress: bool = False,
    ) -> np.ndarray:
        if self.train_data is None:
            raise RuntimeError("Simulation data has not been loaded.")
        if self._subgroup_codes is None or self._language_arr is None:
            raise RuntimeError("Precomputed arrays are not available. Call load() first.")

        subgroup_code = self._vector_to_code(key_vector)

        if language is not None:
            mask = self._language_arr == language
        else:
            mask = np.ones(self._num_rows, dtype=bool)

        if show_progress:
            # Keep the same user-facing option/shape as the original function,
            # even though the heavy work is already precomputed.
            iterator = tqdm(range(len(self.ds_keys)), total=len(self.ds_keys), leave=False)
            for _ in iterator:
                pass

        mask &= self._subgroup_codes == subgroup_code
        return mask

    def process(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.train_data is None or self.all_candidate_idxs is None:
            raise RuntimeError("Simulation data has not been loaded.")
        if (
            self._winner_code is None
            or self._model_a_idx is None
            or self._model_b_idx is None
        ):
            raise RuntimeError("Precomputed arrays are not available. Call load() first.")

        winner_code = self._winner_code[mask]
        valid_mask = (winner_code == 1) | (winner_code == 2)

        winner_code = winner_code[valid_mask]
        a_idx = self._model_a_idx[mask][valid_mask]
        b_idx = self._model_b_idx[mask][valid_mask]

        model_a_wins = winner_code == 1
        winners = np.where(model_a_wins, a_idx, b_idx)
        losers = np.where(model_a_wins, b_idx, a_idx)
        return winners, losers

    def run_subgroup(
        self, key_vector: np.ndarray, language: str | None = None
    ) -> SubgroupResult:
        cache_path = self._subgroup_cache_path(key_vector, language)
        if cache_path.exists():
            cached = np.load(cache_path)
            r_hat = cached["r_hat"]
            if r_hat.size == 0:
                r_hat = None
            misspecification_error = cached["misspecification_error"]
            if misspecification_error.size == 0:
                misspecification_error = None
            else:
                misspecification_error = float(misspecification_error[0])
            return SubgroupResult(
                key_vector=cached["key_vector"],
                winners=cached["winners"],
                losers=cached["losers"],
                r_hat=r_hat,
                misspecification_error=misspecification_error,
            )

        mask = self.filter_fast(key_vector, language=language)
        winners, losers = self.process(mask)
        if len(winners) == 0:
            result = SubgroupResult(
                key_vector=np.asarray(key_vector),
                winners=winners,
                losers=losers,
                r_hat=None,
                misspecification_error=None,
            )
            np.savez_compressed(
                cache_path,
                key_vector=result.key_vector,
                winners=result.winners,
                losers=result.losers,
                r_hat=np.array([], dtype=float),
                misspecification_error=np.array([], dtype=float),
            )
            return result

        r_hat, _ = ut.fit_bradley_terry(winners, losers, self.n_items, beta=self.beta)
        misspec_error = ut.misspecification_error(winners, losers, r_hat, beta=self.beta)

        result = SubgroupResult(
            key_vector=np.asarray(key_vector),
            winners=winners,
            losers=losers,
            r_hat=r_hat,
            misspecification_error=misspec_error,
        )
        np.savez_compressed(
            cache_path,
            key_vector=result.key_vector,
            winners=result.winners,
            losers=result.losers,
            r_hat=result.r_hat,
            misspecification_error=result.misspecification_error,
        )
        return result

    def run(self) -> SimulationResult:
        if self.train_data is None:
            self.load()

        population_cache_path = self._population_cache_path()
        if population_cache_path.exists():
            cached = np.load(population_cache_path, allow_pickle=True)
            misspec_keys = cached["misspec_keys"].tolist()
            misspec_values = cached["misspec_values"].tolist()
            return SimulationResult(
                pop_utilities=cached["pop_utilities"],
                absolute_subsection_sizes=cached["absolute_subsection_sizes"],
                misspec_errors=dict(zip(misspec_keys, misspec_values)),
                included_subgroup_indices=cached["included_subgroup_indices"],
                avg_utils=cached["avg_utils"],
                true_ranking=cached["true_ranking"],
                borda_scores=cached["borda_scores"],
                ranking=cached["ranking"],
                distortion=float(cached["distortion"]),
                k=int(cached["k"]),
            )

        pop_utilities = []
        absolute_subsection_sizes = np.zeros(len(self.vector_keys), dtype=np.int64)
        misspec_errors: dict[int, float] = {}
        included_subgroup_indices: list[int] = []

        for i, vector_key in tqdm(
            enumerate(self.vector_keys),
            total=len(self.vector_keys),
            desc="Computing subgroup fits",
        ):
            subgroup = self.run_subgroup(vector_key, language=None)
            subgroup_size = len(subgroup.winners)
            absolute_subsection_sizes[i] = subgroup_size
            if subgroup_size == 0:
                continue

            pop_utilities.append(subgroup.r_hat)
            misspec_errors[i] = subgroup.misspecification_error
            included_subgroup_indices.append(i)

        if not pop_utilities:
            raise ValueError("No non-empty subgroups were found for the selected configuration.")

        pop_utilities_array = np.stack(pop_utilities)
        included_subgroup_indices_array = np.asarray(included_subgroup_indices, dtype=np.int64)
        voter_dist = None
        if self.weight_voter_dist_by_subgroup_size:
            voter_dist = absolute_subsection_sizes[included_subgroup_indices_array].astype(float)

        avg_utils = pop_utilities_array.sum(axis=0)
        true_ranking = np.argsort(-avg_utils)

        borda_scores, ranking = ut.borda_from_population_utilities(
            pop_utilities_array,
            voter_dist=voter_dist,
            beta=self.beta,
        )
        distortion, k = ut.leaderboard_dist(
            true_ranking=true_ranking,
            ranking=ranking,
            avg_utils=avg_utils,
        )

        np.savez_compressed(
            population_cache_path,
            pop_utilities=pop_utilities_array,
            absolute_subsection_sizes=absolute_subsection_sizes,
            misspec_keys=np.asarray(list(misspec_errors.keys()), dtype=np.int64),
            misspec_values=np.asarray(list(misspec_errors.values()), dtype=float),
            included_subgroup_indices=included_subgroup_indices_array,
            avg_utils=avg_utils,
            true_ranking=true_ranking,
            borda_scores=borda_scores,
            ranking=ranking,
            distortion=distortion,
            k=k,
        )

        return SimulationResult(
            pop_utilities=pop_utilities_array,
            absolute_subsection_sizes=absolute_subsection_sizes,
            misspec_errors=misspec_errors,
            included_subgroup_indices=included_subgroup_indices_array,
            avg_utils=avg_utils,
            true_ranking=true_ranking,
            borda_scores=borda_scores,
            ranking=ranking,
            distortion=distortion,
            k=k,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--keys",
        nargs="+",
        choices=sorted(KEY_NAME_TO_PATH),
        help="Subset of keys to use. Defaults to all keys.",
    )
    parser.add_argument(
        "--weight-voter-dist-by-subgroup-size",
        action="store_true",
        help="Weight subgroup voter mass by subgroup sample counts.",
    )
    parser.add_argument(
        "--wandb-project",
        default="LD",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="jnzhao3",
        help="Optional Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation = DistortionSimulation(
        dataset_name=args.dataset_name,
        split=args.split,
        ds_keys=resolve_ds_keys(args.keys),
        beta=args.beta,
        cache_dir=args.cache_dir,
        weight_voter_dist_by_subgroup_size=args.weight_voter_dist_by_subgroup_size,
    )
    run = None
    run_name = f"simulation_12-{simulation._cache_key_value}"
    wandb_enabled = not args.disable_wandb

    if wandb_enabled:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it or pass --disable-wandb."
            )
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=simulation.config_dict(),
        )
        wandb.alert(
            title=f"{run_name} started",
            text=(
                f"Simulation started on {socket.gethostname()} "
                f"with cache_key={simulation._cache_key_value}."
            ),
        )

    try:
        result = simulation.run()

        print(f"pop_utilities shape: {result.pop_utilities.shape}")
        print(f"distortion: {result.distortion}")
        print(f"k: {result.k}")
        print(
            "misspecification error summary: "
            f"min={min(result.misspec_errors.values())}, "
            f"max={max(result.misspec_errors.values())}"
        )

        if run is not None:
            misspec_table = wandb.Table(
                columns=["subgroup_index", "vector_key", "misspecification_error"]
            )
            subgroup_size_table = wandb.Table(
                columns=["subgroup_index", "vector_key", "subgroup_size"]
            )
            for i, vector_key in enumerate(simulation.vector_keys):
                if i in result.misspec_errors:
                    misspec_table.add_data(
                        i,
                        format_vector_key(vector_key),
                        result.misspec_errors[i],
                    )
                subgroup_size_table.add_data(
                    i,
                    format_vector_key(vector_key),
                    int(result.absolute_subsection_sizes[i]),
                )
            distortion_table = wandb.Table(
                columns=["run_name", "cache_key", "distortion", "k"]
            )
            distortion_table.add_data(
                run_name,
                simulation._cache_key_value,
                result.distortion,
                result.k,
            )

            wandb.log(
                {
                    "distortion": result.distortion,
                    "k": result.k,
                    "num_candidates": len(result.avg_utils),
                    "num_subgroups": len(result.absolute_subsection_sizes),
                    "subgroup_size_min": int(result.absolute_subsection_sizes.min()),
                    "subgroup_size_max": int(result.absolute_subsection_sizes.max()),
                    "subgroup_size_sum": int(result.absolute_subsection_sizes.sum()),
                    "misspec_min": min(result.misspec_errors.values()),
                    "misspec_max": max(result.misspec_errors.values()),
                    "misspec_mean": float(np.mean(list(result.misspec_errors.values()))),
                    "misspecification_by_vector_key": wandb.plot.bar(
                        misspec_table,
                        "vector_key",
                        "misspecification_error",
                        title="Misspecification Error by Vector Key",
                    ),
                    "misspecification_table": misspec_table,
                    "subgroup_size_by_vector_key": wandb.plot.bar(
                        subgroup_size_table,
                        "vector_key",
                        "subgroup_size",
                        title="Subgroup Size by Vector Key",
                    ),
                    "subgroup_size_table": subgroup_size_table,
                    "distortion_summary": wandb.plot.bar(
                        distortion_table,
                        "run_name",
                        "distortion",
                        title="Distortion",
                    ),
                    "distortion_table": distortion_table,
                }
            )
            wandb.run.summary["cache_key"] = simulation._cache_key_value
            wandb.run.summary["population_cache_path"] = str(
                simulation._population_cache_path()
            )
            wandb.run.summary["subgroup_cache_dir"] = str(simulation.cache_dir)
            wandb.alert(
                title=f"{run_name} finished",
                text=(
                    f"Simulation completed with distortion={result.distortion:.6g}, "
                    f"k={result.k}, cache_key={simulation._cache_key_value}."
                ),
            )
    except Exception as exc:
        if run is not None:
            wandb.alert(
                title=f"{run_name} failed",
                text=f"Simulation failed with error: {exc}",
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=1)
        raise
    else:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
