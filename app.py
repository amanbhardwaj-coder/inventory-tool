app.py
import os, csv, io, json, re, argparse
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import pytz
import pandas as pd

# Optional: only needed in Colab terminal mode for Drive upload
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaInMemoryUpload
    from google.colab import auth, files  # type: ignore
    _IN_COLAB = True
except Exception:
    _IN_COLAB = False

# Optional: Streamlit frontend
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False

# ==========================================================
# 0. TIMEZONE
# ==========================================================
def get_ist_now():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

# ==========================================================
# 1. FILE PATHS
# ==========================================================
MAPPING_FILE = "data-headers-2025-10-14.csv"
RULES_FILE = "normalization_rules.json"

# ==========================================================
# 2. HEADER MAPPING
# ==========================================================
VAR_TO_SETTER: Dict[str, str] = {}
SETTER_TO_CANONICAL: Dict[str, str] = {}

IC_TO_BASE = {
    "metal_type_ic": "metals",
    "supported_shapes_ic": "shape",
    "size_ic": "size",
    "center_size_ic": "center_size",
    "ring_mm_ic": "ring_mm",
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def ensure_mapping_file_exists() -> None:
    if not os.path.exists(MAPPING_FILE):
        df = pd.DataFrame([
            {"Setter name": "stock_num", "Header variations": "Stock Number,SKU"},
            {"Setter name": "price", "Header variations": "Price,Base Price,Retail Price,MSRP"},
            {"Setter name": "master_stock", "Header variations": "Master stock,Master Stock,master_stock,masterstock"},
        ])
        df.to_csv(MAPPING_FILE, index=False)

def load_mapping_file() -> None:
    ensure_mapping_file_exists()
    VAR_TO_SETTER.clear()
    SETTER_TO_CANONICAL.clear()

    df_map = pd.read_csv(MAPPING_FILE)
    for _, row in df_map.iterrows():
        setter = str(row.get("Setter name", "")).strip()
        vars_ = [v.strip() for v in str(row.get("Header variations", "")).replace("\r\n", " ").split(",") if v.strip()]
        if setter:
            if vars_:
                SETTER_TO_CANONICAL[setter] = vars_[0]
            for v in vars_:
                VAR_TO_SETTER[_norm(v)] = setter

def update_mapping_manually(header: str, setter: str):
    df = pd.read_csv(MAPPING_FILE)
    valid_setters = df["Setter name"].unique().tolist()
    if setter not in valid_setters:
        raise ValueError(f"Setter '{setter}' not found. Valid: {valid_setters[:20]} ...")
    idx = df[df["Setter name"] == setter].index[0]
    current_vars = str(df.at[idx, "Header variations"])
    if header not in current_vars:
        df.at[idx, "Header variations"] = f"{current_vars}, {header}"
        df.to_csv(MAPPING_FILE, index=False)
        load_mapping_file()

load_mapping_file()

# ==========================================================
# 3. RULES
# ==========================================================
def ensure_rules_file_exists() -> None:
    if not os.path.exists(RULES_FILE):
        base = {
            "version": 1,
            "updated_at": get_ist_now().isoformat(),
            "value_maps": {},
            "global_regex_replacements": []
        }
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)

def load_rules() -> Dict[str, Any]:
    ensure_rules_file_exists()
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================================================
# 4. CSV PARSING / CLEANING
# ==========================================================
def smart_parse(txt: str) -> Dict[str, Any]:
    text = (txt or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    f = io.StringIO(text)
    try:
        dialect = csv.Sniffer().sniff(text[:2000], delimiters=",\t;|")
        delim = dialect.delimiter
    except:
        delim = ","
    reader = csv.reader(f, delimiter=delim)
    rows = [r for r in reader if any((c or "").strip() for c in r)]
    return {
        "columns": [c.strip().replace('"', "") for c in rows[0]] if rows else [],
        "rows": [[(c.strip() if c else None) for c in r] for r in rows[1:]] if rows else []
    }

def get_setter(header: str) -> Optional[str]:
    return VAR_TO_SETTER.get(_norm(header))

def get_canonical(setter: Optional[str], fallback: str) -> str:
    return SETTER_TO_CANONICAL.get(setter, fallback) if setter else fallback

def canonicalize_headers(columns: List[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    rename_map, unknown, seen, new_columns = {}, [], {}, []
    for c in columns:
        setter = get_setter(c)
        if setter:
            base = IC_TO_BASE.get(setter, setter)
            new_name = get_canonical(base, c)
            rename_map[c] = new_name
        else:
            rename_map[c] = c
            unknown.append(c)

    for c in columns:
        nc = rename_map[c]
        seen[nc] = seen.get(nc, 0) + 1
        new_columns.append(f"{nc} ({seen[nc]})" if seen[nc] > 1 else nc)

    return new_columns, {old: new_columns[i] for i, old in enumerate(columns)}, unknown

def clean_input_csv(csv_text: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    parsed = smart_parse(csv_text)
    cols, rows = parsed["columns"], parsed["rows"]
    if not cols:
        return {"cleaned_csv": "", "diff": {}}

    new_cols, rename_map, unknown_cols = canonicalize_headers(cols)

    col_setter_base = [
        IC_TO_BASE.get(get_setter(old), get_setter(old)) if get_setter(old) else None
        for old in cols
    ]

    cleaned_rows = []
    for r in rows:
        rr = []
        for j, v in enumerate(r):
            x = str(v or "").strip().replace(";", ",").replace("|", ",").replace("#", ",")
            x = re.sub(r"\s*,\s*", ",", x)
            if j < len(col_setter_base) and col_setter_base[j]:
                base_key = col_setter_base[j]
                x = rules.get("value_maps", {}).get(base_key, {}).get(x.lower(), x)
            rr.append(x)
        cleaned_rows.append(rr)

    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    w.writerow(new_cols)
    w.writerows(cleaned_rows)

    return {"cleaned_csv": buf.getvalue(), "diff": {"header_renames": rename_map, "unknown_columns": unknown_cols}}

# ==========================================================
# 5. TITLE/DESCRIPTION HELPERS
# ==========================================================
def _strip_available(label: str) -> str:
    return re.sub(r"^available\s+", "", (label or "").strip(), flags=re.IGNORECASE)

def normalize_list(v: str) -> str:
    return ",".join([t.strip() for t in str(v or "").split(",") if t.strip()])

def is_available_col(name: str) -> bool:
    return (name or "").strip().lower().startswith("available ")

def available_base_name(name: str) -> str:
    return _strip_available(name).strip()

def _fmt_ct(val: str) -> str:
    s = (val or "").strip()
    try:
        n = float(s)
        return f"{n:.2f}ct"
    except:
        return s

# ✅ FIX: Canonicalize Shank Style / Head Style (for pricing + sku + image)
def _canon_key(label: str) -> str:
    n = _norm(_strip_available(label))

    if n in ["metals", "metal"]:
        return "Metal"
    if n == "shape":
        return "Shape"
    if n in ["centercaratweight", "caratweight", "centercarat", "centerstone", "centerstonesize"]:
        return "Center Stone"
    if n in ["size", "ringsize"]:
        return "Ring Size"
    if n in ["shankstyle", "shank"]:
        return "Shank Style"
    if n in ["headstyle", "head"]:
        return "Head Style"
    return _strip_available(label)

def _pretty_value(key: str, val: str) -> str:
    v = (val or "").strip()
    k = (key or "").lower()
    if "shape" in k:
        return v.title()
    if "metal" in k:
        return v.title()
    if "center" in k or "carat" in k or "ct" in k:
        return _fmt_ct(v)
    if "stone type" in k:
        return v.title()
    return v

def infer_style_name_from_title(base_title: str, varying_options: List[str]) -> str:
    t = (base_title or "").strip()
    if not t:
        return ""
    opts = sorted({o.strip() for o in varying_options if o and o.strip()}, key=len, reverse=True)
    for o in opts:
        t = re.sub(rf"(?i)\b{re.escape(o)}\b", " ", t)
    t = re.sub(r"\b\d+(\.\d+)?\s*ct\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+(\.\d+)?\s*carat\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*[/,|]\s*", " ", t).strip()
    t = re.sub(r"\s+-\s+", " ", t).strip()
    return t

def _find_first_pos(text: str, needles: List[str]) -> Optional[int]:
    t = (text or "").lower()
    best = None
    for n in needles:
        n2 = (n or "").strip().lower()
        if not n2:
            continue
        m = re.search(rf"\b{re.escape(n2)}\b", t)
        if m:
            best = m.start() if best is None else min(best, m.start())
    return best

def infer_variant_title_order(original_title: str,
                              parts: Dict[str, str],
                              all_options_by_key: Dict[str, List[str]]) -> List[str]:
    title = (original_title or "").strip()
    keys = [k for k in parts.keys() if parts.get(k)]
    scored = []
    for k in keys:
        selected = parts[k]
        needles = [selected] + all_options_by_key.get(k, [])
        pos = _find_first_pos(title, needles)
        scored.append((pos if pos is not None else 10**9, k))
    scored.sort(key=lambda x: x[0])
    return [k for _, k in scored]

def build_variant_short_title(original_title: str,
                              style_name_base: str,
                              parts: Dict[str, str],
                              all_options_by_key: Dict[str, List[str]]) -> str:
    style = (style_name_base or "").strip() or "Item"
    order_keys = infer_variant_title_order(original_title, parts, all_options_by_key)
    front = []
    for k in order_keys:
        v = (parts.get(k) or "").strip()
        if v:
            front.append(v)
    return " ".join(front + [style]).strip() if front else style

def build_variant_description(style_name_base: str,
                              parts: Dict[str, str],
                              all_options_by_key: Dict[str, List[str]],
                              original_title: str) -> str:
    title = build_variant_short_title(original_title, style_name_base, parts, all_options_by_key)
    ordered_keys = infer_variant_title_order(original_title, parts, all_options_by_key)
    lines = []
    for k in ordered_keys:
        if parts.get(k):
            lines.append(f"- {k}: {parts[k]}")
    for k, v in parts.items():
        if k not in ordered_keys and v:
            lines.append(f"- {k}: {v}")
    return (title + "\n\nVariant Details:\n" + "\n".join(lines)).strip()

# ==========================================================
# 6. PRICING (OPT-IN)
# ==========================================================
def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if not s:
        return default
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except:
        return default

def price_rules_enabled(rules: Dict[str, Any]) -> bool:
    pr = (rules or {}).get("price_rules")
    return isinstance(pr, dict) and isinstance(pr.get("adjustments"), dict) and len(pr.get("adjustments")) > 0

def compute_variant_price(base_price_value: Any, parts: Dict[str, str], rules: Dict[str, Any]) -> str:
    pr = rules.get("price_rules", {}) or {}
    base = _to_float(base_price_value, default=_to_float(pr.get("default_base_price", 0), 0.0))
    adjustments = pr.get("adjustments", {}) or {}

    def nk(s: str) -> str:
        return _norm(s or "")

    adj_norm: Dict[str, Dict[str, float]] = {}
    for k, mapping in adjustments.items():
        kk = nk(k)
        adj_norm[kk] = {}
        if isinstance(mapping, dict):
            for val, amt in mapping.items():
                adj_norm[kk][nk(val)] = _to_float(amt, 0.0)

    price = base
    for k, v in (parts or {}).items():
        kk = nk(k)
        vv = nk(v)
        if kk in adj_norm and vv in adj_norm[kk]:
            price += adj_norm[kk][vv]

    return f"{price:.2f}"

# ==========================================================
# 7. IMAGE URLS (OPT-IN)
# ==========================================================
def image_rules_enabled(rules: Dict[str, Any]) -> bool:
    ir = (rules or {}).get("image_rules")
    return isinstance(ir, dict) and ir.get("enabled") is True and bool(ir.get("base_url"))

def _img_safe_token(s: str, upper: bool = True, strip_non_alnum: bool = True) -> str:
    t = str(s or "").strip()
    if strip_non_alnum:
        t = re.sub(r"[^A-Za-z0-9]", "", t)
    return t.upper() if upper else t

def _image_token_for(key: str, value: str, image_rules: Dict[str, Any]) -> str:
    abbr = (image_rules.get("abbr") or {}).get(key) or {}
    if value in abbr:
        return str(abbr[value])
    v_norm = (value or "").strip().lower()
    for k2, v2 in abbr.items():
        if (k2 or "").strip().lower() == v_norm:
            return str(v2)
    fb = image_rules.get("fallback", {}) or {}
    if fb.get("use_raw_if_missing_abbr", True):
        return _img_safe_token(value, upper=fb.get("upper", True), strip_non_alnum=fb.get("strip_non_alnum", True))
    return ""

def generate_image_urls(master_stock: str, parts: Dict[str, str], rules: Dict[str, Any]) -> Dict[str, str]:
    ir = rules.get("image_rules", {}) or {}
    base_url = str(ir.get("base_url") or "")
    ext = str(ir.get("suffix") or "")
    joiner = str(ir.get("joiner") or "_")
    order = ir.get("order") or ["Master stock", "Metal"]
    pos = (ir.get("variant_suffix_position") or "after_ext").strip().lower()

    fb = ir.get("fallback", {}) or {}
    upper = fb.get("upper", True)
    strip_non_alnum = fb.get("strip_non_alnum", True)

    tokens = []
    for k in order:
        if k == "Master stock":
            tokens.append(_img_safe_token(master_stock, upper=upper, strip_non_alnum=strip_non_alnum))
        else:
            if k in parts and parts[k]:
                tokens.append(_image_token_for(k, parts[k], ir))

    filename_base = joiner.join([t for t in tokens if t])

    out = {}
    variants = ir.get("variants") or []
    if variants:
        for v in variants:
            col = v.get("column")
            if not col:
                continue
            prefix = str(v.get("path_prefix") or "")
            suf = str(v.get("path_suffix") or "")
            fname = filename_base + suf + ext if pos == "before_ext" else filename_base + ext + suf
            out[col] = base_url + prefix + fname
        return out

    for i in range(1, 5):
        suf = f"_{i}"
        fname = filename_base + suf + ext if pos == "before_ext" else filename_base + ext + suf
        out[f"Image URL {i}"] = base_url + fname
    return out

# ==========================================================
# 8. SKU SHORTENING (OPT-IN)
# ==========================================================
def sku_rules_enabled(rules: Dict[str, Any]) -> bool:
    sr = (rules or {}).get("sku_rules")
    return isinstance(sr, dict) and sr.get("enabled", True) is True

def shorten_sku(master_stock: str, parts: Dict[str, str], rules: Dict[str, Any]) -> str:
    sr = (rules or {}).get("sku_rules", {}) or {}
    joiner = str(sr.get("joiner") or "-")
    fallback_max_len = int(sr.get("fallback_max_len") or 8)
    abbr_map = sr.get("abbr", {}) or {}

    preferred_order = sr.get("order")
    if not isinstance(preferred_order, list) or not preferred_order:
        preferred_order = sorted(parts.keys())

    keys_in_order = [k for k in preferred_order if k in parts and parts.get(k)]
    remaining = sorted([k for k in parts.keys() if k not in keys_in_order and parts.get(k)])
    final_keys = keys_in_order + remaining

    def abbr_token(key: str, val: str) -> str:
        val = (val or "").strip()
        if not val:
            return ""
        d = abbr_map.get(key, {}) if isinstance(abbr_map.get(key, {}), dict) else {}
        if val in d:
            return str(d[val]).strip()
        v_norm = val.lower().strip()
        for k2, v2 in d.items():
            if str(k2).lower().strip() == v_norm:
                return str(v2).strip()
        tok = re.sub(r"[^A-Za-z0-9]", "", val).upper()
        return tok[:fallback_max_len] if tok else ""

    tokens = [str(master_stock).strip()]
    for k in final_keys:
        t = abbr_token(k, parts.get(k, ""))
        if t:
            tokens.append(t)

    seen = set()
    out = []
    for t in tokens:
        if t and t not in seen:
            out.append(t)
            seen.add(t)

    return joiner.join(out)

# ==========================================================
# 9. OUTPUT COLUMN ORDERING (Base + Available side-by-side)
# ==========================================================
def build_ordered_headers(final_rows: List[Dict[str, str]],
                          original_cols: List[str],
                          has_price_column: bool,
                          include_images: bool) -> List[str]:
    """
    Goal:
    - Start with Master/Stock/Title/Desc/Price/Images
    - Then list attributes in a grouped way:
      Base column immediately followed by its "Available <Base>" column (if present)
    - Preserve original column order as much as possible.
    """
    priority = ["Master stock", "Stock Number", "Short Title", "Description"]
    if has_price_column:
        priority.append("Price")
    if include_images:
        for i in range(1, 5):
            priority.append(f"Image URL {i}")

    # Determine what keys exist at all
    all_keys = set()
    for r in final_rows:
        all_keys.update(r.keys())

    # Helper: detect base/available pairs
    def is_avail(k: str) -> bool:
        return str(k).strip().lower().startswith("available ")

    def base_of_available(k: str) -> str:
        return _strip_available(k).strip()

    # Seed with priority
    out_h: List[str] = []
    for h in priority:
        if h in all_keys and h not in out_h:
            out_h.append(h)

    # Build preferred attribute order:
    # Start from original cols (preserves their order), but expand into (base, available base) pairs.
    visited = set(out_h)

    def add_col(c: str):
        if c in all_keys and c not in visited:
            out_h.append(c)
            visited.add(c)

    for c in original_cols:
        if c in priority:
            continue

        c_str = str(c or "").strip()
        if not c_str:
            continue

        # If it's an Available col, add base then available (if base exists)
        if is_avail(c_str):
            base = base_of_available(c_str)
            add_col(base)
            add_col(c_str)
            continue

        # If it's a base col and it has an Available partner, add base then partner
        avail_partner = f"Available {c_str}"
        if avail_partner in all_keys:
            add_col(c_str)
            add_col(avail_partner)
        else:
            add_col(c_str)

    # Finally add any remaining columns not yet included (stable sorted for determinism)
    remaining = [k for k in sorted(all_keys) if k not in visited]
    for k in remaining:
        add_col(k)

    return out_h

# ==========================================================
# 10. EXPAND INVENTORY
# ==========================================================
def expand_inventory(csv_text: str, rules: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (expanded_csv, meta)
      meta includes diff-like info you can show in UI.
    """
    if rules is None:
        rules = load_rules()

    data = smart_parse(csv_text)
    cols, rows = list(data["columns"]), [list(r) for r in data["rows"]]
    if not rows:
        return "", {"error": "No rows found."}

    SKIP_EXPANSION_COLS = {
        "short title", "description",
        "price",
        "image url 1", "image url 2", "image url 3", "image url 4",
        "image_url_1", "image_url_2", "image_url_3", "image_url_4"
    }

    def _h(s: str) -> str:
        return _norm(s or "")

    MASTER_INPUT_CANDIDATES = {"master stock", "masterstock", "master_stock"}
    master_idx = next((i for i, c in enumerate(cols) if _h(c) in MASTER_INPUT_CANDIDATES), -1)
    if master_idx == -1:
        raise ValueError("❌ Input file must contain a 'Master stock' column.")

    price_idx = next((i for i, c in enumerate(cols) if _h(c) == _h("Price")), -1)
    has_price_column = (price_idx != -1)

    img_norm_set = {_norm(f"Image URL {i}") for i in range(1, 5)}
    img_cols_present = any(_norm(c) in img_norm_set for c in cols)
    include_images = (img_cols_present or ((not img_cols_present) and image_rules_enabled(rules)))

    stock_col_name = next((c for c in cols if _h(c) in {_h("Stock Number"), _h("SKU")}), "Stock Number")

    # variation flags: commas mean multiple tokens
    v_flags = []
    for i in range(len(cols)):
        if (cols[i] or "").strip().lower() in SKIP_EXPANSION_COLS:
            v_flags.append(False)
        else:
            v_flags.append(any(len(str(r[i]).split(",")) > 1 for r in rows if i < len(r) and r[i]))

    final_rows: List[Dict[str, str]] = []
    master_out_h = "Master stock"
    stock_out_h = "Stock Number"

    for i, row in enumerate(rows):
        master_val = str(row[master_idx] or "").strip()
        if not master_val:
            master_val = f"MASTER-{i+1:03}"

        base_price_value = row[price_idx] if has_price_column and price_idx < len(row) else None

        original_title = str(row[cols.index("Short Title")] or "").strip() if "Short Title" in cols else ""
        original_desc  = str(row[cols.index("Description")] or "").strip() if "Description" in cols else ""

        exp_meta = []
        for idx in range(len(cols)):
            col_name = cols[idx]
            if (col_name or "").strip().lower() in SKIP_EXPANSION_COLS:
                continue

            raw_val = str(row[idx] or "")
            raw_val_norm = raw_val.replace("#", ",")
            tokens = [x.strip() for x in raw_val_norm.split(",") if x.strip()] or [""]

            exp_meta.append({
                "col": col_name,
                "tokens": tokens,
                "orig": raw_val_norm,
                "varies": v_flags[idx],
                "is_available": is_available_col(col_name),
                "available_base": available_base_name(col_name) if is_available_col(col_name) else "",
                "idx": idx
            })

        varying_options: List[str] = []
        for meta in exp_meta:
            if meta["varies"]:
                varying_options.extend([x.strip() for x in str(meta["orig"] or "").split(",") if x.strip()])
        style_name_base = infer_style_name_from_title(original_title, varying_options) or original_title or master_val

        all_options_by_key: Dict[str, List[str]] = {}
        for meta in exp_meta:
            if meta["varies"]:
                label = meta["available_base"] if meta["is_available"] else _strip_available(meta["col"])
                key = _canon_key(label)
                opts = [x.strip() for x in str(meta["orig"] or "").split(",") if x.strip()]
                all_options_by_key[key] = [_pretty_value(key, o) for o in opts]

        for combo in product(*[x["tokens"] for x in exp_meta]):
            new_r: Dict[str, str] = {}
            for idx, c in enumerate(cols):
                new_r[c] = str(row[idx] or "").strip()

            new_r[master_out_h] = master_val

            parts: Dict[str, str] = {}

            for meta, token in zip(exp_meta, combo):
                token = (token or "").strip()

                if meta["is_available"]:
                    base_name = meta["available_base"]
                    new_r[f"Available {base_name}"] = normalize_list(meta["orig"])
                    new_r[base_name] = token

                    if meta["varies"] and token:
                        key = _canon_key(base_name)
                        parts[key] = _pretty_value(key, token)
                    continue

                out_col = meta["col"]
                new_r[out_col] = token

                if meta["varies"] and token:
                    key = _canon_key(out_col)
                    parts[key] = _pretty_value(key, token)

            # Stock Number
            existing_sku = str(new_r.get(stock_col_name, "")).strip()
            if existing_sku:
                new_r[stock_out_h] = existing_sku
            else:
                new_r[stock_out_h] = shorten_sku(master_val, parts, rules) if sku_rules_enabled(rules) else master_val

            # Titles
            new_r["Short Title"] = build_variant_short_title(original_title, style_name_base, parts, all_options_by_key)
            new_r["Description"] = build_variant_description(style_name_base, parts, all_options_by_key, original_title)

            # Normalize Available*
            for k in list(new_r.keys()):
                if str(k).strip().lower().startswith("available "):
                    new_r[k] = normalize_list(str(new_r[k] or "").replace("#", ","))

            # Price
            if has_price_column:
                if price_rules_enabled(rules):
                    new_r["Price"] = compute_variant_price(base_price_value, parts, rules)
                else:
                    new_r["Price"] = str(base_price_value or "").strip()

            # Images
            if (not img_cols_present) and image_rules_enabled(rules):
                img_map = generate_image_urls(master_stock=master_val, parts=parts, rules=rules)
                for k, v in img_map.items():
                    new_r[k] = v

            final_rows.append(new_r)

    # Build ordered headers (Base + Available side-by-side)
    out_h = build_ordered_headers(final_rows, cols, has_price_column, include_images)

    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    w.writerow(out_h)
    w.writerows([[r.get(h, "") for h in out_h] for r in final_rows])

    return buf.getvalue(), {
        "rows_out": len(final_rows),
        "has_price_column": has_price_column,
        "images_generated": (not img_cols_present) and image_rules_enabled(rules),
        "sku_shortened": sku_rules_enabled(rules)
    }

# ==========================================================
# 11. COLAB DRIVE SAVE (optional)
# ==========================================================
def save_to_drive(prefix: str, content: str, folder_id: str, is_csv: bool = False) -> str:
    if not _IN_COLAB:
        raise RuntimeError("save_to_drive is only available in Colab environment.")
    drive = build("drive", "v3")
    ext = "csv" if is_csv else "json"
    fname = f"{prefix}_{get_ist_now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/csv" if is_csv else "application/json")
    res = drive.files().create(
        body={"name": fname, "parents": [folder_id]},
        media_body=media,
        fields="id, webViewLink",
        supportsAllDrives=True
    ).execute()
    return res.get("webViewLink", "N/A")

# ==========================================================
# 12. STREAMLIT FRONTEND
# ==========================================================
def run_streamlit_app():
    if not _HAS_STREAMLIT:
        raise RuntimeError("Streamlit is not installed. Run: pip install streamlit")

    st.set_page_config(page_title="Inventory Expander", layout="wide")
    st.title("Inventory Expander (No LLM)")

    st.sidebar.header("Rules")
    st.sidebar.write(f"Rules file: `{RULES_FILE}`")
    rules = load_rules()
    st.sidebar.json(rules)

    st.markdown("Upload a CSV with `Master stock` and any variation columns (comma-separated).")

    up = st.file_uploader("Upload CSV", type=["csv", "txt"])
    if not up:
        st.stop()

    raw = up.read().decode("utf-8", errors="ignore")

    # Clean step (shows unknown headers)
    c = clean_input_csv(raw, rules)
    st.subheader("Header Cleaning Diff")
    st.write("Unknown columns (not mapped):", c["diff"].get("unknown_columns", []))
    st.write("Header renames:", c["diff"].get("header_renames", {}))

    if st.button("Run Expansion"):
        expanded, meta = expand_inventory(c["cleaned_csv"], rules=rules)
        st.success(f"Done. Rows out: {meta['rows_out']}")
        st.json(meta)

        st.download_button(
            label="Download Expanded CSV",
            data=expanded.encode("utf-8"),
            file_name="expanded.csv",
            mime="text/csv"
        )

        st.subheader("Preview (first 30 lines)")
        st.code("\n".join(expanded.splitlines()[:30]))

# ==========================================================
# 13. TERMINAL MODE (Colab-friendly)
# ==========================================================
def run_terminal_mode():
    print("🛠️ TERMINAL MODE (NO LLM)")
    rules = load_rules()

    while True:
        print("\n" + "="*40)
        print("1) Upload CSV and Expand")
        print("2) Force Map Header (edit mapping csv)")
        print("3) Exit")
        choice = input("> ").strip()

        if choice == "1":
            if not _IN_COLAB:
                path = input("Enter local CSV path: ").strip()
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            else:
                up = files.upload()
                if not up:
                    continue
                raw = list(up.values())[0].decode("utf-8", errors="ignore")

            c = clean_input_csv(raw, rules)
            expanded, meta = expand_inventory(c["cleaned_csv"], rules=rules)
            print(f"✅ Expanded rows: {meta['rows_out']}")
            print("Preview:\n", "\n".join(expanded.splitlines()[:10]))

        elif choice == "2":
            h = input("Header to fix: ").strip()
            s = input("Setter name (existing): ").strip()
            try:
                update_mapping_manually(h, s)
                print("✅ Mapping updated.")
            except Exception as e:
                print("❌", e)

        elif choice == "3":
            print("Goodbye!")
            break

# ==========================================================
# 14. ENTRYPOINT
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["terminal", "streamlit"], default="terminal")
    args, _ = parser.parse_known_args()

    if args.mode == "streamlit":
        run_streamlit_app()
    else:
        run_terminal_mode()

if __name__ == "__main__":
    main()
