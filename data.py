import os
import json
import math
import random
from typing import Tuple, Optional, List
import numpy as np

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)

def _sigmf_complex_dtype_from_meta(meta_dict: dict):
    dt = (meta_dict.get('global', {}) or {}).get('core:datatype', None)
    if dt is None:
        return np.complex128
    dt = dt.lower()
    if dt == 'cf32_le':
        return np.complex64
    if dt == 'cf64_le':
        return np.complex128
    raise ValueError(f'Unsupported SigMF core:datatype = {dt}. Extend mapping if needed.')

def _read_sigmf_iq_as_memmap(prefix_path: str):
    meta_path = f'{prefix_path}.sigmf-meta'
    data_path = f'{prefix_path}.sigmf-data'
    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        raise FileNotFoundError(f'Missing SigMF files: {meta_path} or {data_path}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    c_dtype = _sigmf_complex_dtype_from_meta(meta_dict)
    file_size = os.path.getsize(data_path)
    itemsize = np.dtype(c_dtype).itemsize
    n_complex = file_size // itemsize
    iq = np.memmap(data_path, dtype=c_dtype, mode='r', shape=(n_complex,))
    return (iq, meta_dict)

def wifi_dataset_slice_sigmf(base_dir: str, ft: int, runs: Tuple[int, ...], classi: List[int], iq_len: int, max_segments_per_file: Optional[int]=None):
    devicename = ['3123D7B', '3123D7D', '3123D7E', '3123D52', '3123D54', '3123D58', '3123D64', '3123D65', '3123D70', '3123D76', '3123D78', '3123D79', '3123D80', '3123D89', '3123EFE', '3124E4A']
    X_all = np.zeros((0, 2, iq_len), dtype=np.float32)
    y_all = np.zeros((0,), dtype=np.int64)
    target = 0
    for cls in classi:
        device_id = devicename[cls]
        for r in runs:
            prefix = os.path.join(base_dir, f'{ft}ft', f'WiFi_air_X310_{device_id}_{ft}ft_run{r}')
            (iq, _) = _read_sigmf_iq_as_memmap(prefix)
            I = np.real(iq)
            Q = np.imag(iq)
            n_seg = math.floor(len(I) / iq_len)
            if max_segments_per_file is not None:
                n_seg = min(n_seg, int(max_segments_per_file))
            if n_seg <= 0:
                continue
            X = np.zeros((n_seg, 2, iq_len), dtype=np.float32)
            y = np.full((n_seg,), fill_value=target, dtype=np.int64)
            for k in range(n_seg):
                b = k * iq_len
                X[k, 0, :] = I[b:b + iq_len]
                X[k, 1, :] = Q[b:b + iq_len]
            X_all = np.concatenate([X_all, X], axis=0)
            y_all = np.concatenate([y_all, y], axis=0)
        target += 1
    return (X_all, y_all)

def _select_k_per_class_shuffled(X, y, num_classes: int, k: int, seed: int):
    rng = np.random.RandomState(seed)
    idx_all = []
    for c in range(num_classes):
        idc = np.where(y == c)[0]
        if len(idc) == 0:
            continue
        idc = idc.copy()
        rng.shuffle(idc)
        idx_all.extend(idc[:min(len(idc), k)].tolist())
    idx_all = np.array(idx_all, dtype=np.int64)
    return (X[idx_all], y[idx_all])

def stratified_split_three_way(X: np.ndarray, y: np.ndarray, num_classes: int, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    (tr_idx, va_idx, te_idx) = ([], [], [])
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        if len(idx) < 3:
            tr_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * val_ratio)))
        n_test = max(1, int(round(len(idx) * test_ratio)))
        while n_val + n_test >= len(idx):
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        va_idx.extend(idx[:n_val].tolist())
        te_idx.extend(idx[n_val:n_val + n_test].tolist())
        tr_idx.extend(idx[n_val + n_test:].tolist())
    tr_idx = np.array(tr_idx, dtype=np.int64)
    va_idx = np.array(va_idx, dtype=np.int64)
    te_idx = np.array(te_idx, dtype=np.int64)
    return (X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X[te_idx], y[te_idx])

def power_normalize_segments(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    if X.size == 0:
        return X
    p = np.mean(X[:, 0, :] ** 2 + X[:, 1, :] ** 2, axis=1)
    scale = np.sqrt(p + eps).astype(np.float32)
    Xn = X / scale[:, None, None]
    return Xn.astype(np.float32)

def zscore_with_train_stats(X_train: np.ndarray, *others):
    if X_train.size == 0:
        return (X_train,) + others
    mean = float(X_train.mean())
    std = float(X_train.std())
    if std <= 0:
        std = 1.0
    X_train_n = (X_train - mean) / (std + 1e-08)
    out = [X_train_n.astype(np.float32)]
    for X in others:
        if X is None or (isinstance(X, np.ndarray) and X.size == 0):
            out.append(X)
        else:
            out.append(((X - mean) / (std + 1e-08)).astype(np.float32))
    return tuple(out)

def get_oracle_run1_run2_splits(base_dir: str, iq_len: int=2048, ft: int=2, run1: Tuple[int, ...]=(1,), run2: Tuple[int, ...]=(2,), num_classes: int=16, seed: int=42, val_ratio: float=0.1, test_ratio: float=0.1, max_segments_per_file: Optional[int]=None, k_run1_total_per_class: Optional[int]=1000, k_run2_per_class: Optional[int]=1000, use_power_norm: bool=True, use_zscore: bool=True):
    set_seed(seed)
    classi = list(range(num_classes))
    (X1, y1) = wifi_dataset_slice_sigmf(base_dir=base_dir, ft=ft, runs=run1, classi=classi, iq_len=iq_len, max_segments_per_file=max_segments_per_file)
    if k_run1_total_per_class is not None:
        (X1, y1) = _select_k_per_class_shuffled(X1, y1, num_classes, int(k_run1_total_per_class), seed=seed)
    (X_tr, y_tr, X_va, y_va, X_te, y_te) = stratified_split_three_way(X1, y1, num_classes=num_classes, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    (X2, y2) = wifi_dataset_slice_sigmf(base_dir=base_dir, ft=ft, runs=run2, classi=classi, iq_len=iq_len, max_segments_per_file=max_segments_per_file)
    if k_run2_per_class is not None:
        (X2, y2) = _select_k_per_class_shuffled(X2, y2, num_classes, int(k_run2_per_class), seed=seed)
    if use_power_norm:
        X_tr = power_normalize_segments(X_tr)
        X_va = power_normalize_segments(X_va)
        X_te = power_normalize_segments(X_te)
        X2 = power_normalize_segments(X2)
    if use_zscore:
        (X_tr, X_va, X_te, X2) = zscore_with_train_stats(X_tr, X_va, X_te, X2)
    return ((X_tr, y_tr.astype(np.int64)), (X_va, y_va.astype(np.int64)), (X_te, y_te.astype(np.int64)), (X2, y2.astype(np.int64)))
