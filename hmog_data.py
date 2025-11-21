import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# 按照 HMOG 预处理产出的字段顺序保留 12 组传感器特征
FEATURE_COLUMNS: Tuple[str, ...] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "acc_magnitude",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "gyr_magnitude",
    "mag_x",
    "mag_y",
    "mag_z",
    "mag_magnitude",
)

# 需要遍历的时间窗口（秒）
WINDOW_SIZES: Tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
)

SAMPLE_RATE = 100  # 10ms 采样周期
DEFAULT_OVERLAP = 0.5
CACHE_VERSION = "v1"


def list_available_users(base_path: Path) -> List[str]:
    """Return sorted user ids (directory names) under the processed HMOG path."""
    return sorted([p.name for p in base_path.iterdir() if p.is_dir()])


def _load_split_df(base_path: Path, user_id: str, split: str) -> pd.DataFrame:
    file_path = base_path / user_id / f"{user_id}_{split}_normalized.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing split file: {file_path}")
    df = pd.read_csv(file_path)
    df = df.sort_values(["session", "timestamp"]).reset_index(drop=True)
    return df


def _df_to_session_arrays(df: pd.DataFrame) -> List[np.ndarray]:
    """Group dataframe by session and return feature arrays per session."""
    session_arrays: List[np.ndarray] = []
    for _, group in df.groupby("session"):
        session_arrays.append(group[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32))
    return session_arrays


def load_all_user_splits(base_path: Path, user_ids: Sequence[str]) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Load train/val/test session arrays for each user."""
    cache: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for user_id in user_ids:
        cache[user_id] = {}
        for split in ("train", "val", "test"):
            df = _load_split_df(base_path, user_id, split)
            cache[user_id][split] = _df_to_session_arrays(df)
    return cache


def load_user_session_splits(base_path: Path, user_id: str) -> Dict[str, List[np.ndarray]]:
    """Load a single user's train/val/test session arrays."""
    return {
        split: _df_to_session_arrays(_load_split_df(base_path, user_id, split))
        for split in ("train", "val", "test")
    }


def _cache_file_path(cache_dir: Path, user_id: str, window_size_sec: float, overlap: float, target_width: int) -> Path:
    ws_tag = str(window_size_sec).replace(".", "p")
    return cache_dir / f"user_{user_id}_ws_{ws_tag}_ov{int(overlap * 100)}_tw{target_width}_{CACHE_VERSION}.npz"


def load_cached_user_windows(
    cache_dir: Path, user_id: str, window_size_sec: float, overlap: float, target_width: int
) -> Dict[str, np.ndarray]:
    """Load cached window tensors for a user if present."""
    cache_path = _cache_file_path(cache_dir, user_id, window_size_sec, overlap, target_width)
    if not cache_dir or not cache_path.exists():
        return {}

    data = np.load(cache_path)
    return {"train": data["train"], "val": data["val"], "test": data["test"]}


def save_cached_user_windows(
    cache_dir: Path,
    user_id: str,
    window_size_sec: float,
    overlap: float,
    target_width: int,
    windows: Dict[str, np.ndarray],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_path(cache_dir, user_id, window_size_sec, overlap, target_width)
    np.savez_compressed(
        cache_path,
        train=windows["train"],
        val=windows["val"],
        test=windows["test"],
        meta=json.dumps(
            {
                "user": user_id,
                "window_size": window_size_sec,
                "overlap": overlap,
                "target_width": target_width,
                "version": CACHE_VERSION,
            }
        ),
    )
    return cache_path


def ensure_cached_user_windows(
    cache_dir: Path,
    user_id: str,
    window_size_sec: float,
    overlap: float,
    target_width: int,
    prep_workers: int,
    session_cache: Dict[str, List[np.ndarray]] = None,
    base_path: Path = None,
) -> Dict[str, np.ndarray]:
    """
    Return cached windows for a user; if absent, build and persist them.
    """
    if cache_dir:
        cached = load_cached_user_windows(cache_dir, user_id, window_size_sec, overlap, target_width)
        if cached:
            return cached

    if session_cache is None:
        if base_path is None:
            raise ValueError("base_path or session_cache must be provided to build windows.")
        session_cache = load_user_session_splits(base_path, user_id)

    windows = {
        split: build_windows(session_cache[split], window_size_sec, overlap, target_width, prep_workers)
        for split in ("train", "val", "test")
    }

    if cache_dir:
        save_cached_user_windows(cache_dir, user_id, window_size_sec, overlap, target_width, windows)
    return windows


def _precompute_single_user_window(args: Tuple[str, str, str, float, float, int, int, int]) -> Tuple[str, float, str]:
    """
    Worker entrypoint for concurrent cache building.

    Args tuple: (base_path, cache_dir, user_id, window_size, overlap, target_width, prep_workers, process_workers)
    """
    base_path, cache_dir, user_id, window_size, overlap, target_width, prep_workers, process_workers = args
    base_path = Path(base_path)
    cache_dir = Path(cache_dir)

    # Avoid spawning another large process pool inside every worker; keep inner workers modest.
    inner_workers = max(1, min(prep_workers, max(mp.cpu_count() // max(process_workers, 1), 1)))
    session_cache = load_user_session_splits(base_path, user_id)
    ensure_cached_user_windows(
        cache_dir=cache_dir,
        user_id=user_id,
        window_size_sec=window_size,
        overlap=overlap,
        target_width=target_width,
        prep_workers=inner_workers,
        session_cache=session_cache,
    )
    resolved_path = _cache_file_path(cache_dir, user_id, window_size, overlap, target_width)
    return user_id, window_size, str(resolved_path)


def precompute_all_user_windows(
    base_path: Path,
    cache_dir: Path,
    users: Sequence[str],
    window_sizes: Sequence[float],
    overlap: float,
    target_width: int,
    prep_workers: int,
    process_workers: int = 40,
) -> List[Tuple[str, float, str]]:
    """
    Precompute and cache windowed tensors for all (user, window_size) pairs.
    Returns list of tuples (user_id, window_size, cache_path).
    """
    tasks: List[Tuple[str, str, str, float, float, int, int, int]] = []
    for user_id in users:
        for ws in window_sizes:
            tasks.append(
                (str(base_path), str(cache_dir), user_id, float(ws), overlap, target_width, prep_workers, process_workers)
            )

    if not tasks:
        return []

    effective_workers = min(process_workers, len(tasks), mp.cpu_count() or process_workers)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, float, str]] = []
    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp.get_context("spawn")) as ex:
        for user_id, window_size, path in ex.map(_precompute_single_user_window, tasks):
            results.append((user_id, window_size, path))
    return results


def _session_windows(session: np.ndarray, window_len: int, step: int) -> List[np.ndarray]:
    windows: List[np.ndarray] = []
    if session.shape[0] < window_len:
        return windows
    for start in range(0, session.shape[0] - window_len + 1, step):
        segment = session[start : start + window_len].T  # (features, window)
        windows.append(segment)
    return windows


def build_windows(
    session_arrays: Sequence[np.ndarray],
    window_size_sec: float,
    overlap: float = DEFAULT_OVERLAP,
    target_width: int = 50,
    num_workers: int = 0,
) -> np.ndarray:
    """Generate sliding windows for a list of session arrays."""
    window_len = max(int(round(window_size_sec * SAMPLE_RATE)), 1)
    step = max(int(window_len * (1 - overlap)), 1)

    def process_session(session: np.ndarray) -> List[np.ndarray]:
        local_windows: List[np.ndarray] = []
        for segment in _session_windows(session, window_len, step):
            if target_width and segment.shape[1] != target_width:
                # 将不同窗口长度线性插值到统一长度，保证解码器输出尺寸匹配
                time_steps = np.linspace(0, segment.shape[1] - 1, num=target_width)
                segment = np.stack(
                    [np.interp(time_steps, np.arange(segment.shape[1]), ch) for ch in segment], axis=0
                )
            local_windows.append(segment)
        return local_windows

    all_windows: List[np.ndarray] = []
    if num_workers and num_workers > 1:
        # 优先使用多进程池并行不同 session，以占满 CPU 核心；若受限则退回线程池
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for local in ex.map(process_session, session_arrays):
                    if local:
                        all_windows.extend(local)
        except Exception:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                for local in ex.map(process_session, session_arrays):
                    if local:
                        all_windows.extend(local)
    else:
        for session in session_arrays:
            local = process_session(session)
            if local:
                all_windows.extend(local)

    if not all_windows:
        return np.zeros((0, 1, len(FEATURE_COLUMNS), target_width), dtype=np.float32)

    stacked = np.stack(all_windows)
    stacked = np.expand_dims(stacked, axis=1)  # (N, 1, features, window)
    return stacked.astype(np.float32, copy=False)


class WindowedHMOGDataset(Dataset):
    """Torch dataset that returns (window_tensor, label)."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = windows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]


def prepare_user_datasets(
    target_user: str,
    window_size_sec: float,
    cache: Dict[str, Dict[str, List[np.ndarray]]] = None,
    overlap: float = DEFAULT_OVERLAP,
    target_width: int = 50,
    prep_workers: int = 0,
    max_negative_per_split: int = None,
    negative_users: Sequence[str] = None,
    max_eval_per_split: int = None,
    window_cache_dir: Path = None,
    base_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build window tensors and labels for一个目标用户。

    Args:
        target_user: 当前训练/评估的用户 id。
        window_size_sec: 时间窗口（秒）。
        cache: 通过 load_all_user_splits 构建的 session 缓存。
        overlap: 窗口重叠率。
        target_width: 将窗口重采样到统一的时间步长，保持解码器输出尺寸一致。
        max_negative_per_split: 限制验证/测试集的负样本数量，避免爆内存。
        negative_users: 指定用作攻击样本的用户列表，默认除目标用户外的所有用户。
        window_cache_dir: 指定缓存目录时，将优先读取/写入缓存，避免重复切窗。
        base_path: 当未提供 cache 时用于临时读取用户 session。

    Returns:
        train_x, train_y, val_x, val_y, test_x, test_y
    """

    cache_dir = Path(window_cache_dir) if window_cache_dir else None
    base_path = Path(base_path) if base_path else base_path

    def user_windows(user_id: str) -> Dict[str, np.ndarray]:
        session_cache = cache[user_id] if cache and user_id in cache else None
        return ensure_cached_user_windows(
            cache_dir=cache_dir,
            user_id=user_id,
            window_size_sec=window_size_sec,
            overlap=overlap,
            target_width=target_width,
            prep_workers=prep_workers,
            session_cache=session_cache,
            base_path=base_path,
        )

    pos_windows = user_windows(target_user)
    pos_train = pos_windows["train"]
    pos_val = pos_windows["val"]
    pos_test = pos_windows["test"]

    if negative_users is not None:
        neg_candidates = [uid for uid in negative_users if uid != target_user]
    elif cache:
        neg_candidates = [uid for uid in cache.keys() if uid != target_user]
    else:
        raise ValueError("negative_users must be provided when cache is None.")

    neg_val_list: List[np.ndarray] = []
    neg_test_list: List[np.ndarray] = []

    for uid in neg_candidates:
        neg_windows = user_windows(uid)
        neg_val_list.append(neg_windows["val"])
        neg_test_list.append(neg_windows["test"])

    neg_val = np.concatenate(neg_val_list, axis=0) if neg_val_list else np.zeros_like(pos_val)
    neg_test = np.concatenate(neg_test_list, axis=0) if neg_test_list else np.zeros_like(pos_test)

    if max_negative_per_split is not None:
        if len(neg_val) > max_negative_per_split:
            idx = np.random.choice(len(neg_val), size=max_negative_per_split, replace=False)
            neg_val = neg_val[idx]
        if len(neg_test) > max_negative_per_split:
            idx = np.random.choice(len(neg_test), size=max_negative_per_split, replace=False)
            neg_test = neg_test[idx]

    train_x = pos_train
    train_y = np.ones(len(train_x), dtype=np.float32)

    val_x = np.concatenate([pos_val, neg_val], axis=0)
    val_y = np.concatenate([np.ones(len(pos_val), dtype=np.float32), np.zeros(len(neg_val), dtype=np.float32)], axis=0)

    test_x = np.concatenate([pos_test, neg_test], axis=0)
    test_y = np.concatenate([np.ones(len(pos_test), dtype=np.float32), np.zeros(len(neg_test), dtype=np.float32)], axis=0)

    if max_eval_per_split is not None:
        if len(val_x) > max_eval_per_split:
            idx = np.random.choice(len(val_x), size=max_eval_per_split, replace=False)
            val_x, val_y = val_x[idx], val_y[idx]
        if len(test_x) > max_eval_per_split:
            idx = np.random.choice(len(test_x), size=max_eval_per_split, replace=False)
            test_x, test_y = test_x[idx], test_y[idx]

    return train_x, train_y, val_x, val_y, test_x, test_y
