from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from . import MetricContext, register_metric


def _ensure_plausibility_path() -> None:
    root = Path(__file__).resolve().parents[2] / "submodules" / "plausibility"
    if root.exists() and str(root) not in sys.path:
        sys.path.append(str(root))


_ensure_plausibility_path()
try:
    from plausibility.autoregressor import AutoRegressor as PMAutoRegressor
except Exception as exc:  # pragma: no cover
    PMAutoRegressor = None
    _PLAUSIBILITY_IMPORT_ERROR = str(exc)
else:
    _PLAUSIBILITY_IMPORT_ERROR = None


def _create_bin_info_if_missing(dataset: str, real_data_path: str) -> None:
    base_dir = Path("datasets") / dataset
    bin_info_path = base_dir / "bin_info.json"
    if bin_info_path.exists():
        return

    try:
        df = pd.read_csv(real_data_path)
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        bin_info = {col: 10 for col in numerical_cols} if numerical_cols else {}
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(bin_info_path, "w", encoding="utf-8") as f:
            json.dump(bin_info, f, indent=4)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to generate bin_info.json: {exc}")


def _dataset_from_real_path(real_data_path: str) -> Optional[str]:
    if not real_data_path:
        return None
    path = Path(real_data_path)
    if path.name.endswith("_train.csv"):
        return path.name.replace("_train.csv", "")
    if path.name == "train.csv" and path.parent.name:
        return path.parent.name
    return None


def _assign_category(value, bins):
    if pd.isna(value):
        for bin_range in bins:
            if bin_range.get("lower") is None and bin_range.get("upper") is None:
                return bin_range["category"]
    else:
        for bin_range in bins:
            lower = bin_range.get("lower")
            upper = bin_range.get("upper")
            if lower is not None and upper is not None and lower <= value < upper:
                return bin_range["category"]
        numeric_bins = sorted([b for b in bins if b.get("lower") is not None], key=lambda x: x["lower"])
        if numeric_bins and value == numeric_bins[-1]["upper"]:
            return numeric_bins[-1]["category"]
    return 0


def _train_model_if_missing(dataset: str, epochs: int = 50, batch_size: int = 128) -> bool:
    base_dir = Path("datasets") / dataset
    model_dir = base_dir / "plausibility_model"
    model_path = model_dir / "best_model.pt"
    mapping_path = model_dir / "mapping_and_bins.json"
    if model_path.exists() and mapping_path.exists():
        return True

    cmd = [
        "python",
        str(Path("submodules") / "plausibility" / "plausibility" / "tp.py"),
        "--dataname",
        dataset,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--assume-yes",
        "--auto-bins",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        print(f"Plausibility training failed\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}")
        return False
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected plausibility training error: {exc}")
        return False
    return model_path.exists() and mapping_path.exists()


def compute_plausibility_for_paths(
    real_data_path: str,
    generated_samples_path: str,
    train_if_missing: bool = False,
    train_epochs: int = 50,
) -> dict:
    if PMAutoRegressor is None:
        return {"plausibility_avg": None, "error": _PLAUSIBILITY_IMPORT_ERROR}

    try:
        dataset = _dataset_from_real_path(real_data_path)
        if not dataset:
            return {"plausibility_avg": None}

        cache_dir = Path("expdir") / f"output_{dataset}"
        cache_path = cache_dir / "plausibility.csv"
        if cache_path.exists():
            cache_df = pd.read_csv(cache_path)
            cached_row = cache_df[cache_df["file_path"] == generated_samples_path]
            if not cached_row.empty:
                return {"plausibility_avg": float(cached_row["avg_plausibility"].iloc[0])}

        _create_bin_info_if_missing(dataset, real_data_path)

        base_dir = Path("datasets") / dataset
        model_dir = base_dir / "plausibility_model"
        model_path = model_dir / "best_model.pt"
        mapping_path = model_dir / "mapping_and_bins.json"
        if not (model_path.exists() and mapping_path.exists()):
            if train_if_missing:
                ok = _train_model_if_missing(dataset, epochs=train_epochs)
                if not ok:
                    return {"plausibility_avg": None}
            else:
                return {"plausibility_avg": None}

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        col_id_mapping = mapping.get("col_id_mapping", {})
        bin_info = mapping.get("bin_info", {})

        syn = pd.read_csv(generated_samples_path)
        syn = syn.map(lambda x: x.strip() if isinstance(x, str) else x)

        binned = syn.copy()
        for col, bins in bin_info.items():
            if col in binned.columns:
                binned[col] = binned[col].apply(lambda x: _assign_category(x, bins))

        enc = binned.copy()
        fallback_ids = {col: next(iter(mapping.values())) for col, mapping in col_id_mapping.items() if mapping}
        for col in enc.columns:
            if col in col_id_mapping:
                mapping_dict = col_id_mapping[col]
                default_id = fallback_ids.get(col, 0)

                if pd.api.types.is_numeric_dtype(binned[col]):

                    def _map_numeric(v):
                        if pd.isna(v):
                            return default_id
                        try:
                            return mapping_dict.get(int(v), default_id)
                        except (ValueError, TypeError):
                            return default_id

                    enc[col] = binned[col].map(_map_numeric)
                else:
                    enc[col] = binned[col].astype(str).map(lambda v: mapping_dict.get(v, default_id))
        for col in enc.columns:
            if col in col_id_mapping:
                enc[col] = enc[col].fillna(fallback_ids.get(col, 0))
            else:
                enc[col] = enc[col].fillna(0)
        if enc.isna().any().any():
            enc = enc.fillna(0)

        x = torch.from_numpy(enc.astype("int64").values)

        md_vocab = mapping.get("vocab_size")
        md_dmodel = mapping.get("d_model")

        def _resolve_hidden_size() -> int:
            if md_dmodel is not None:
                return int(md_dmodel)
            try:
                return AutoConfig.from_pretrained("bert-base-uncased", local_files_only=True).hidden_size
            except Exception:
                try:
                    return AutoConfig.from_pretrained("bert-base-uncased").hidden_size
                except Exception as exc:
                    print(f"Unable to load bert-base-uncased config ({exc}); defaulting to 768.")
                    return 768

        d_model = _resolve_hidden_size()
        vocab_size = int(md_vocab) if md_vocab is not None else (int(enc.max().max()) + 1 if enc.size > 0 else 2000)

        model = PMAutoRegressor(vocab_size, d_model)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            batch = 128
            total_loss = 0.0
            total_count = 0
            for i in range(0, x.shape[0], batch):
                inp = x[i : i + batch]
                loss = model(inp)
                bs = inp.shape[0]
                total_loss += loss.item() * bs
                total_count += bs
        avg = float(total_loss / total_count) if total_count > 0 else None

        if avg is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            row = pd.DataFrame([{"file_path": generated_samples_path, "avg_plausibility": avg}])
            if cache_path.exists():
                row.to_csv(cache_path, mode="a", header=False, index=False)
            else:
                row.to_csv(cache_path, index=False)

        return {"plausibility_avg": avg}
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {"plausibility_avg": None, "error": str(exc)}


@register_metric("plausibility")
def compute_plausibility_metric(ctx: MetricContext) -> dict:
    cfg = ctx.settings.raw_config.get("plausibility_metrics", {})
    train_if_missing = cfg.get("train_if_missing", False)
    train_epochs = cfg.get("train_epochs", 50)
    return compute_plausibility_for_paths(
        ctx.settings.real_data_path,
        ctx.synthetic_path,
        train_if_missing=train_if_missing,
        train_epochs=train_epochs,
    )
