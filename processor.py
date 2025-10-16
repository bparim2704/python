#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

# ==============================
# Configuration and constants
# ==============================

# Width of one denomination block (columns per block)
BLOCK_WIDTH = 6

# Known semantic labels that can appear inside blocks
SEMANTIC_LABELS = {
    "Grand", "Super", "Major", "Max", "Mor", "Mini",
    "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
    "FG", "ITS ALIVE"
}

# Symbols considered base anchors (their immediate right neighbor should be numeric in a well-formed block)
BASE_LABELS = {"Max", "Mor", "Mini", "g", "h", "i", "d", "e", "f", "a", "b", "c"}

# Multipliers allowed for derived detection
DERIVED_MULTIPLIERS = list(range(1, 11))  # 1×..10×

# Preferred canonical denomination per group (used if present; otherwise we fall back dynamically)
# credit_low covers 1CT/2CT, credit_hi covers 5CT/10CT
PREFERRED_GROUP_CANONICAL = {
    "credit_low": ("1CT", 0.60),
    "credit_hi": ("5CT", 1.50),
    "$1": ("$1", 10.0),
    "$2": ("$2", 20.0),
}

# Families and simple detection helpers
CREDIT_TOKENS = {"1CT", "2CT", "5CT", "10CT"}
DOLLAR_TOKENS = {"$1", "$2"}

# Explicit family map and scale factors (used as hints and for validation/fallback)
FAMILY_MAP: Dict[str, str] = {
    # credit family
    "1CT $0.60": "credit", "1CT $1.20": "credit", "1CT $1.80": "credit",
    "1CT $3": "credit", "1CT $6": "credit",
    "2CT $1.20": "credit", "2CT $2.40": "credit", "2CT $3.60": "credit",
    "2CT $6": "credit", "2CT $12": "credit",
    "5CT $1.50": "credit", "5CT $3": "credit", "5CT $4.50": "credit",
    "5CT $7.50": "credit", "5CT $15": "credit",
    "10CT $3": "credit", "10CT $6": "credit", "10CT $9": "credit",
    "10CT $15": "credit", "10CT $30": "credit",

    # $1 family
    "$1 $5": "$1", "$1 $10": "$1", "$1 $15": "$1", "$1 $20": "$1", "$1 $25": "$1",

    # $2 family
    "$2 $10": "$2", "$2 $20": "$2", "$2 $30": "$2", "$2 $40": "$2", "$2 $50": "$2",
}

SCALE_FACTORS: Dict[str, float] = {
    # credit family (relative to 1CT $0.60)
    "1CT $0.60": 1, "1CT $1.20": 2, "1CT $1.80": 3, "1CT $3": 5, "1CT $6": 10,
    "2CT $1.20": 2, "2CT $2.40": 4, "2CT $3.60": 6, "2CT $6": 10, "2CT $12": 20,
    "5CT $1.50": 2.5, "5CT $3": 5, "5CT $4.50": 7.5, "5CT $7.50": 12.5, "5CT $15": 25,
    "10CT $3": 5, "10CT $6": 10, "10CT $9": 15, "10CT $15": 25, "10CT $30": 50,

    # $1 family (relative to $1 $10)
    "$1 $10": 1, "$1 $5": 0.5, "$1 $15": 1.5, "$1 $20": 2, "$1 $25": 2.5,

    # $2 family (relative to $2 $20)
    "$2 $20": 1, "$2 $10": 0.5, "$2 $30": 1.5, "$2 $40": 2, "$2 $50": 2.5,
}

# Row → label mapping used for baseline construction and validation
BASE_ROW_LABELS: Dict[int, List[str]] = {
    3: ["Max", "Mor", "Mini"],
    4: ["g", "h", "i"],
    5: ["d", "e", "f"],
    6: ["a", "b", "c"],
}

# Optional baselines for specific canonical product keys (minimal, scalable).
# Values may be strings with $/commas; they are parsed via clean_number.
BASELINE_ROWS_BY_KEY_RAW: Dict[str, Dict[int, List[Any]]] = {
    "1CT $0.60": {
        3: ["$30", "$15", "$10"],     # cash
        4: [1000, 500, 300],              # credits
        5: [200, 150, 125],               # credits
        6: [100, 75, 60],                 # credits
    },
    "$1 $10": {
        3: ["$500", "$250", "$150"],   # cash
        4: ["$150", "$80", "$50"],
        5: ["$40", "$30", "$20"],
        6: ["$16", "$12", "$10"],
    },
    "$2 $20": {
        3: ["$1,000", "$500", "$300"], # cash
        4: ["$300", "$160", "$100"],
        5: ["$80", "$60", "$40"],
        6: ["$32", "$24", "$20"],
    },
}

# Curated overrides where scaling deviates from simple denom ratio
BASELINE_ROWS_BY_KEY_RAW.update({
    "5CT $3": {
        3: ["$150", "$80", "$50"],
        4: [1000, 500, 300],
        5: [200, 150, 120],
        6: [100, 80, 60],
    },
    "10CT $3": {
        3: ["$150", "$80", "$50"],
        4: [500, 250, 150],
        5: [100, 75, 60],
        6: [50, 40, 30],
    },
})


def _rows_scale(rows: Dict[int, List[float]], factor: float) -> Dict[int, List[float]]:
    return {r: [round(v * factor, 6) for v in vs] for r, vs in rows.items()}


def family_of_label_token(token: str) -> str:
    token = token.strip()
    if any(token.startswith(ct) for ct in CREDIT_TOKENS) or "CT" in token:
        return "credit"
    if token.startswith("$1"):
        return "$1"
    if token.startswith("$2"):
        return "$2"
    return "unknown"


def _family_of_key(full_key: str) -> str:
    tok = full_key.split(" ")[0] if full_key else ""
    return family_of_label_token(tok)


def clean_number(val: Any) -> Optional[float]:
    s = str(val).strip().replace(",", "")
    if not s or s in {"·", "nan"}:
        return None
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except Exception:
        return None


def _normalize_baseline_rows(raw: Dict[str, Dict[int, List[Any]]]) -> Dict[str, Dict[int, List[float]]]:
    norm: Dict[str, Dict[int, List[float]]] = {}
    for key, rows in raw.items():
        nr: Dict[int, List[float]] = {}
        for r, vals in rows.items():
            parsed = []
            for v in vals:
                num = clean_number(v)
                if num is None:
                    continue
                parsed.append(float(num))
            if parsed:
                nr[r] = parsed
        if nr:
            norm[key] = nr
    return norm


BASELINE_ROWS_BY_KEY: Dict[str, Dict[int, List[float]]] = _normalize_baseline_rows(BASELINE_ROWS_BY_KEY_RAW)


# ==============================
# Data classes
# ==============================

@dataclass
class DenomBlock:
    key: str                     # e.g., "1CT $0.60"
    family: str                  # "credit" | "$1" | "$2" | "unknown"
    denom_value: float           # numeric denomination value (e.g., 0.60, 10, 20.0)
    start_col: int               # inclusive column index in the full DF
    end_col: int                 # inclusive column index in the full DF
    semantics: Dict[Tuple[int, int], str]
    base_values: Dict[str, Dict[str, float]]  # {label: {credits, cash}}
    derived: Dict[Tuple[int, int], Dict[str, Any]]


# ==============================
# Utilities
# ==============================


def is_numeric_str(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    s = s.replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    return bool(re.fullmatch(r"\d+(\.\d+)?", s))


class _IatAccessor:
    def __init__(self, grid: "SimpleGrid") -> None:
        self._grid = grid

    def __getitem__(self, key: Tuple[int, int]) -> Any:
        r, c = key
        return self._grid.data[r][c]

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        r, c = key
        self._grid.data[r][c] = value


class _RowView:
    def __init__(self, row: List[Any]) -> None:
        self._row = row

    def tolist(self) -> List[Any]:
        return list(self._row)


class _ILocAccessor:
    def __init__(self, grid: "SimpleGrid") -> None:
        self._grid = grid

    def __getitem__(self, r: int) -> _RowView:
        return _RowView(self._grid.data[r])


class SimpleGrid:
    def __init__(self, table: List[List[Any]]) -> None:
        if not table:
            table = []
        width = max((len(r) for r in table), default=0)
        self.data: List[List[Any]] = [list(r) + [""] * (width - len(r)) for r in table]
        self.iat = _IatAccessor(self)
        self.iloc = _ILocAccessor(self)

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.data), len(self.data[0]) if self.data else 0)


def to_df(table: List[List[Any]]) -> SimpleGrid:
    """Build a SimpleGrid from a ragged list-of-lists without treating any row as header."""
    return SimpleGrid(table)


def slice_blocks(df: Any, block_width: int = BLOCK_WIDTH) -> List[Tuple[int, int]]:
    """Return (start_col_idx, end_col_idx) for each block across the DF columns."""
    blocks = []
    ncols = df.shape[1]
    for start in range(0, ncols, block_width):
        end = min(start + block_width - 1, ncols - 1)
        if start <= end:
            blocks.append((start, end))
    return blocks


def parse_block_denom(df: Any, start: int, end: int) -> Tuple[str, float, Tuple[int, int]]:
    """Scan a block to find a denomination token (1CT/$1/$2/etc.) and its numeric value to the right."""
    for r in range(df.shape[0]):
        for c in range(start, end + 1):
            cell = str(df.iat[r, c]).strip()
            if not cell:
                continue
            if any(cell.startswith(t) for t in CREDIT_TOKENS | DOLLAR_TOKENS):
                nxt = str(df.iat[r, c + 1]).strip() if c + 1 <= end else ""
                val = clean_number(nxt) or 0.0
                return f"{cell} {nxt}".strip(), val, (r, c)
    return "", 0.0, (-1, -1)


def detect_semantics(df: Any, start: int, end: int) -> Dict[Tuple[int, int], str]:
    semantics: Dict[Tuple[int, int], str] = {}
    for r in range(df.shape[0]):
        for c in range(start, end + 1):
            val = str(df.iat[r, c]).strip()
            if val in SEMANTIC_LABELS or val in BASE_LABELS:
                semantics[(r, c)] = val
    return semantics


def extract_base_values(df: Any, start: int, end: int, denom_value: float, family: str) -> Dict[str, Dict[str, float]]:
    base: Dict[str, Dict[str, float]] = {}
    for r in range(df.shape[0]):
        for c in range(start, end):
            label = str(df.iat[r, c]).strip()
            if label in BASE_LABELS:
                right_raw = str(df.iat[r, c + 1]).strip()
                if is_numeric_str(right_raw) or right_raw.startswith("$"):
                    val = clean_number(right_raw)
                    if val is None:
                        continue
                    if family in {"$1", "$2"}:
                        # For $ families, grid numbers are already cash; we mirror into credits for multiplier logic
                        base[label] = {"credits": float(val), "cash": float(val)}
                    else:
                        base[label] = {"credits": float(val), "cash": round(float(val) * denom_value, 6)}
    return base


def detect_derived_values(
    df: Any,
    start: int,
    end: int,
    base_values: Dict[str, Dict[str, float]],
    denom_value: float,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    derived: Dict[Tuple[int, int], Dict[str, Any]] = {}

    if not base_values:
        return derived

    for r in range(df.shape[0]):
        for c in range(start, end + 1):
            raw = str(df.iat[r, c]).strip()
            if not raw:
                continue

            # Cash overlays (e.g., "$600"): preserve and attempt to map to base values
            if raw.startswith("$") and is_numeric_str(raw):
                cash_val = clean_number(raw)
                if cash_val is not None:
                    # Try to find a clean integer multiple or direct match against base cash values
                    candidates: List[Tuple[int, float, str]] = []  # (factor, base_cash, symbol)
                    for sym, bv in base_values.items():
                        base_cash = float(bv.get("cash") or 0.0)
                        if base_cash <= 0:
                            continue
                        if math.isclose(cash_val, base_cash, rel_tol=1e-6, abs_tol=1e-2):
                            candidates.append((1, base_cash, sym))
                            continue
                        m = cash_val / base_cash
                        if abs(m - round(m)) < 0.01:
                            candidates.append((int(round(m)), base_cash, sym))
                    derived_from = "overlay"
                    if candidates:
                        # prefer lower multiplier, then lower base_cash
                        candidates.sort(key=lambda x: (x[0], x[1]))
                        k, base_cash, sym = candidates[0]
                        derived_from = (f"{k}×{sym}" if k != 1 else sym)
                    derived[(r, c)] = {
                        "credits": None,
                        "cash": float(cash_val),
                        "derived_from": derived_from,
                    }
                continue

            # Numeric credits/cash
            if not is_numeric_str(raw):
                continue
            val = float(clean_number(raw) or 0.0)

            matched = False
            for sym, bv in base_values.items():
                base_cr = float(bv.get("credits") or 0.0)
                if base_cr == 0.0:
                    continue
                for m in DERIVED_MULTIPLIERS:
                    if math.isclose(val, base_cr * m, rel_tol=1e-6, abs_tol=1e-6):
                        derived[(r, c)] = {
                            "credits": val,
                            "cash": round(val * denom_value, 6),
                            "derived_from": f"{m}×{sym}" if m != 1 else sym,
                        }
                        matched = True
                        break
                if matched:
                    break
    return derived


# ==============================
# Canonical base selection per family
# ==============================


def choose_canonical_block(blocks: List["DenomBlock"]) -> Optional["DenomBlock"]:
    if not blocks:
        return None

    family = blocks[0].family

    # Split credit into two subgroups: low (1CT/2CT) vs high (5CT/10CT)
    if family == "credit":
        low = [b for b in blocks if any(b.key.startswith(tok) for tok in ("1CT", "2CT"))]
        hi = [b for b in blocks if any(b.key.startswith(tok) for tok in ("5CT", "10CT"))]
        # Prefer low group if present, else high group
        group_blocks = low if low else hi
        group_key = "credit_low" if low else "credit_hi"
        pref_token, pref_value = PREFERRED_GROUP_CANONICAL.get(group_key, (None, None))
    else:
        # $1 / $2 families use direct preferences
        pref_token, pref_value = PREFERRED_GROUP_CANONICAL.get(family, (None, None))

    # 1) Prefer exact token + closest to preferred denom value
    if pref_token is not None:
        candidates = [b for b in blocks if b.key.startswith(pref_token)]
        if candidates:
            # Choose the one whose denom_value is closest to preferred
            if pref_value is not None:
                best = min(candidates, key=lambda b: abs(b.denom_value - pref_value))
                return best
            # else arbitrary among candidates
            return candidates[0]

    # 2) Otherwise choose the one with the smallest denom_value (good default for credits)
    try:
        return min(blocks, key=lambda b: (b.denom_value if b.denom_value > 0 else float("inf")))
    except ValueError:
        return blocks[0]


# ==============================
# Overlay band mapping (dynamic, anchored by header rows)
# ==============================


def overlay_pairs_dynamic(df: Any) -> List[Tuple[range, range]]:
    """
    Build overlay (base_rows -> overlay_rows) pairs dynamically using "Grand" header rows as anchors.
    For each anchor row r, the base rows are r..r+5 and overlays are r+6..r+11.
    This matches: 7–12 overlay 1–6; 20–25 overlay 14–19; 32–37 overlay 26–31; 45–50 overlay 39–44; ...
    """
    pairs: List[Tuple[range, range]] = []
    r = 0
    total_rows = df.shape[0]
    while r + 11 < total_rows:
        row_has_grand = any(str(x).strip() == "Grand" for x in df.iloc[r].tolist())
        if row_has_grand:
            base = range(r + 0, r + 6)
            over = range(r + 6, r + 12)
            pairs.append((base, over))
            r += 12
        else:
            r += 1
    return pairs


# ==============================
# Block extraction and scaling
# ==============================


def get_baseline_rows_for_key(product_key: str) -> Optional[Dict[int, List[float]]]:
    """Return baseline rows for a product_key, scaling from a known baseline if needed."""
    if product_key in BASELINE_ROWS_BY_KEY:
        return BASELINE_ROWS_BY_KEY[product_key]

    fam = _family_of_key(product_key)
    if fam == "unknown":
        return None

    # Try to find any baseline in same family we can scale from using SCALE_FACTORS
    if product_key in SCALE_FACTORS:
        target_rel = SCALE_FACTORS[product_key]
        # Prefer a natural canonical per family
        preferred = {
            "credit": "1CT $0.60",
            "$1": "$1 $10",
            "$2": "$2 $20",
        }.get(fam)
        candidates = []
        for base_key in BASELINE_ROWS_BY_KEY.keys():
            if _family_of_key(base_key) != fam:
                continue
            if base_key in SCALE_FACTORS:
                candidates.append(base_key)
        # Try preferred first if present, else any candidate
        base_key = preferred if preferred in candidates else (candidates[0] if candidates else None)
        if base_key:
            base_rel = SCALE_FACTORS.get(base_key, 1.0)
            rows = BASELINE_ROWS_BY_KEY[base_key]
            if base_rel:
                factor = target_rel / base_rel
                return _rows_scale(rows, factor)
    return None


def build_base_from_baseline(product_key: str, denom_value: float) -> Optional[Dict[str, Dict[str, float]]]:
    rows = get_baseline_rows_for_key(product_key)
    if not rows:
        return None
    fam = _family_of_key(product_key)
    base_values: Dict[str, Dict[str, float]] = {}
    for row_idx, labels in BASE_ROW_LABELS.items():
        values = rows.get(row_idx)
        if not values or len(values) != 3:
            continue
        for i, label in enumerate(labels):
            v = float(values[i])
            if fam == "credit":
                if row_idx == 3:
                    # row 3 is cash; convert to credits using denom_value
                    credits = (v / denom_value) if denom_value else v
                    base_values[label] = {"credits": round(credits, 6), "cash": round(v, 6)}
                else:
                    # rows 4..6 are credits
                    base_values[label] = {"credits": round(v, 6), "cash": round(v * denom_value, 6)}
            else:
                # $ families treat all as cash amounts; store as credits numerically for multiplier logic
                base_values[label] = {"credits": round(v, 6), "cash": round(v, 6)}
    return base_values if base_values else None


@dataclass
class _BlockInfo:
    block: DenomBlock
    info: Dict[str, Any]


def extract_blocks(df: Any) -> List[DenomBlock]:
    blocks: List[DenomBlock] = []
    for start, end in slice_blocks(df):
        key, denom, _ = parse_block_denom(df, start, end)
        fam = family_of_label_token(key.split(" ")[0]) if key else "unknown"
        semantics = detect_semantics(df, start, end)
        base_vals = extract_base_values(df, start, end, denom, fam)
        # If table is missing bases, attempt to build from baselines/scales
        if not base_vals and key:
            baseline_built = build_base_from_baseline(key, denom)
            if baseline_built:
                base_vals = baseline_built
        derived_vals = detect_derived_values(df, start, end, base_vals, denom)
        blocks.append(
            DenomBlock(
                key=key or f"unknown_{start}_{end}",
                family=fam,
                denom_value=float(denom or 0.0),
                start_col=start,
                end_col=end,
                semantics=semantics,
                base_values=base_vals,
                derived=derived_vals,
            )
        )
    return blocks


def scale_from_canonical(df: Any, blocks: List[DenomBlock]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping keyed by block.key:
      - canonical_base key
      - scale_factor (relative to canonical denom_value)
      - semantic_template (positions from canonical)
      - regenerated base_values (scaled credits + recomputed cash)
      - derived (detected from values present in each block using regenerated bases)
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    # Group by family
    fam_to_blocks: Dict[str, List[DenomBlock]] = {}
    for b in blocks:
        fam_to_blocks.setdefault(b.family, []).append(b)

    for fam, fam_blocks in fam_to_blocks.items():
        # Find canonical for the family
        canonical = choose_canonical_block(fam_blocks)
        if not canonical:
            continue

        c_val = canonical.denom_value if canonical.denom_value else 1.0
        c_sem = canonical.semantics
        c_base = canonical.base_values if canonical.base_values else build_base_from_baseline(canonical.key, canonical.denom_value) or {}

        # If canonical block lacks bases (e.g., empty), keep empty; we'll just annotate semantics
        for b in fam_blocks:
            # Use explicit SCALE_FACTORS if keys are recognized; else fall back to denom ratio
            if canonical.key in SCALE_FACTORS and b.key in SCALE_FACTORS:
                # scale from canonical by dividing their relative factors
                canon_rel = SCALE_FACTORS.get(canonical.key, 1.0)
                blk_rel = SCALE_FACTORS.get(b.key, b.denom_value / c_val if c_val else 1.0)
                scale = float(blk_rel) / float(canon_rel) if canon_rel else 1.0
            else:
                scale = (b.denom_value / c_val) if c_val else 1.0

            # Regenerate/scale base credits from canonical base_values
            regen_base: Dict[str, Dict[str, float]] = {}
            if c_base:
                for sym, vals in c_base.items():
                    cr = float(vals.get("credits", 0.0))
                    sc_cr = round(cr * scale, 6)
                    # For $ families, treat credits as cash
                    if fam in {"$1", "$2"}:
                        regen_base[sym] = {"credits": sc_cr, "cash": sc_cr}
                    else:
                        regen_base[sym] = {"credits": sc_cr, "cash": round(sc_cr * b.denom_value, 6)}

            # Detect derived inside this block using regenerated bases
            derived_vals = detect_derived_values(
                df, b.start_col, b.end_col, regen_base if regen_base else b.base_values, b.denom_value
            )

            mapping[b.key] = {
                "family": fam,
                "denom_value": b.denom_value,
                "canonical_base": canonical.key,
                "scale_factor": round(scale, 6),
                "semantic_template": c_sem if c_sem else b.semantics,
                "base_values": regen_base if regen_base else b.base_values,
                "derived": derived_vals if derived_vals else b.derived,
            }

    return mapping


# ==============================
# In-place update (labels + numbers) with overlays handled gracefully
# ==============================


def write_block_labels_and_numbers(
    df: Any,
    block: DenomBlock,
    info: Dict[str, Any],
) -> None:
    start, end = block.start_col, block.end_col

    # 1) Ensure semantic labels from canonical template are present in this block
    for (r, c), label in info.get("semantic_template", {}).items():
        if start <= c <= end and 0 <= r < df.shape[0]:
            cell = str(df.iat[r, c]).strip()
            # Only write label if cell is empty-ish to avoid clobbering explicit content
            if cell in {"", "·", "nan"}:
                df.iat[r, c] = label

    # 2) For each base label found in this block, write scaled credits to the immediate right cell
    base_vals = info.get("base_values", {})
    if base_vals:
        for r in range(df.shape[0]):
            for c in range(start, end):
                label = str(df.iat[r, c]).strip()
                if label in base_vals and (c + 1) <= end:
                    df.iat[r, c + 1] = base_vals[label]["credits"]

    # 3) Overlays: replicate numbers into overlay rows when a clean multiplier is present,
    #    and also mirror base right-of-label numbers when overlay slots are empty.
    for base_rng, over_rng in overlay_pairs_dynamic(df):
        for i, r_base in enumerate(base_rng):
            if r_base >= df.shape[0]:
                continue
            r_over = over_rng[i] if i < len(over_rng) else None
            if r_over is None or r_over >= df.shape[0]:
                continue

            for c in range(start, end + 1):
                base_cell = str(df.iat[r_base, c]).strip()
                over_cell = str(df.iat[r_over, c]).strip()

                # Mirror base label+number into overlay when both overlay cells are empty-ish
                if base_cell in base_vals and (c + 1) <= end:
                    over_right = str(df.iat[r_over, c + 1]).strip()
                    if over_cell in {"", "·", "nan"} and over_right in {"", "·", "nan"}:
                        df.iat[r_over, c] = base_cell
                        df.iat[r_over, c + 1] = base_vals[base_cell]["credits"]
                    continue

                # Mirror bare numeric values into overlay when overlay slot is empty-ish
                if is_numeric_str(base_cell):
                    if over_cell in {"", "·", "nan"}:
                        df.iat[r_over, c] = base_cell
                    continue

                # Propagate xN multipliers into overlay row using nearest base label to the left
                if base_cell.startswith("x") and re.fullmatch(r"x(\d+)", base_cell):
                    try:
                        mult = int(base_cell[1:])
                    except ValueError:
                        mult = None
                    if mult is None or mult <= 0:
                        continue

                    # Find nearest base label to the left in the same base row and use its credits
                    nearest_label: Optional[str] = None
                    for lc in range(c - 1, start - 1, -1):
                        lbl = str(df.iat[r_base, lc]).strip()
                        if lbl in base_vals:
                            nearest_label = lbl
                            break

                    if nearest_label and (c + 1) <= end:
                        base_cr = base_vals[nearest_label]["credits"]
                        # Only write into overlay if empty-ish (respect notes/FG/cash overlays)
                        if over_cell in {"", "·", "nan"}:
                            df.iat[r_over, c] = base_cell
                        over_right = str(df.iat[r_over, c + 1]).strip()
                        if over_right in {"", "·", "nan"}:
                            df.iat[r_over, c + 1] = round(float(base_cr) * mult, 6)


# ==============================
# End-to-end processing
# ==============================


def process_dataframe(df: Any) -> Any:
    # SimpleGrid does not define .empty; use shape checks
    if df.shape[0] == 0 or df.shape[1] == 0:
        return df

    # Extract all blocks first
    blocks = extract_blocks(df)

    # Build scaling info from per-family canonical selection
    info_by_key = scale_from_canonical(df, blocks)

    # Write updates block-by-block (in input order)
    for b in blocks:
        info = info_by_key.get(b.key)
        if not info:
            # No canonical info; still try to write existing bases/semantics for this block only
            info = {
                "semantic_template": b.semantics,
                "base_values": b.base_values,
            }
        write_block_labels_and_numbers(df, b, info)

    return df


# ==============================
# CSV I/O
# ==============================


def load_csv(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def save_csv(path: str, df: SimpleGrid) -> None:
    # Save with no header and no index to preserve original shape-like output
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in df.data:
            writer.writerow(row)


# ==============================
# CLI
# ==============================


def main(inputs_dir: str = "inputs", output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(inputs_dir):
        os.makedirs(inputs_dir, exist_ok=True)
        print("Created inputs/ directory. Place CSV files inside and rerun.")
        return

    any_found = False
    for name in sorted(os.listdir(inputs_dir)):
        if not name.lower().endswith(".csv"):
            continue
        in_path = os.path.join(inputs_dir, name)
        try:
            table = load_csv(in_path)
            df = to_df(table)
            updated = process_dataframe(df)
            out_path = os.path.join(output_dir, f"processed_{name}")
            save_csv(out_path, updated)
            print(f"Processed {name} -> {out_path}")
            any_found = True
        except Exception as e:
            print(f"Failed {name}: {e}")

    if not any_found:
        print("No CSV files found in inputs/. Nothing to do.")


if __name__ == "__main__":
    main()
