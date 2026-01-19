import os, csv, io, json, re, argparse
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import pytz
import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False

# ==========================================================
# 0. TIMEZONE & CONFIG
# ==========================================================
if _HAS_STREAMLIT:
    st.set_page_config(page_title="Inventory Expander Pro", layout="wide")

def get_ist_now():
    try:
        ist = pytz.timezone("Asia/Kolkata")
        return datetime.now(ist)
    except:
        return datetime.now()

# ==========================================================
# 1. FILE PATHS & PERSISTENCE
# ==========================================================
MAPPING_FILE = "data-headers.csv"
RULES_FILE = "normalization_rules.json"

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

def ensure_files_exist() -> None:
    # 1. Header Mapping File
    if not os.path.exists(MAPPING_FILE):
        df = pd.DataFrame([
            {"Setter name": "stock_num", "Header variations": "Stock Number,SKU,Style#"},
            {"Setter name": "price", "Header variations": "Price,Base Price,Retail Price,MSRP"},
            {"Setter name": "master_stock", "Header variations": "Master stock,Master Stock,master_stock,masterstock,Group ID"},
        ])
        df.to_csv(MAPPING_FILE, index=False)

    # 2. Rules File
    if not os.path.exists(RULES_FILE):
        base = {
            "version": 1,
            "updated_at": get_ist_now().isoformat(),
            "value_maps": {
                 "metals": {"y": "Yellow Gold", "w": "White Gold", "r": "Rose Gold"},
                 "shape": {"rnd": "Round", "ov": "Oval"}
            },
            "sku_rules": {"enabled": True, "joiner": "-", "fallback_max_len": 8},
            "image_rules": {"enabled": False, "base_url": "https://example.com/images/", "suffix": ".jpg"},
            "price_rules": {"currency": "USD", "default_base_price": 0, "adjustments": {}},
        }
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)

def load_mapping_file() -> None:
    ensure_files_exist()
    VAR_TO_SETTER.clear()
    SETTER_TO_CANONICAL.clear()
    try:
        df_map = pd.read_csv(MAPPING_FILE)
        for _, row in df_map.iterrows():
            setter = str(row.get("Setter name", "")).strip()
            vars_ = [v.strip() for v in str(row.get("Header variations", "")).replace("\r\n", " ").split(",") if v.strip()]
            if setter:
                if vars_:
                    SETTER_TO_CANONICAL[setter] = vars_[0]
                for v in vars_:
                    VAR_TO_SETTER[_norm(v)] = setter
    except:
        pass

def load_rules() -> Dict[str, Any]:
    ensure_files_exist()
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_rules(rules: Dict[str, Any]) -> None:
    rules = dict(rules or {})
    rules["updated_at"] = get_ist_now().isoformat()
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

load_mapping_file()

# ==========================================================
# 2. VISUAL EDITOR HELPERS
# ==========================================================
def rules_to_dataframe(rules):
    """Converts JSON value_maps to a DataFrame for the UI."""
    rows = []
    maps = rules.get("value_maps", {})
    for category, mappings in maps.items():
        for k, v in mappings.items():
            rows.append({"Type": "Rename", "Category": category, "Input (CSV)": k, "Output (Final)": v})
    return pd.DataFrame(rows)

def dataframe_to_rules(df, base_rules):
    """Updates the value_maps in base_rules from the UI DataFrame."""
    new_maps = {}
    if not df.empty and "Type" in df.columns:
        rename_df = df[df["Type"] == "Rename"]
        for _, row in rename_df.iterrows():
            cat = str(row.get("Category", "")).strip().lower()
            inp = str(row.get("Input (CSV)", "")).strip().lower()
            out = str(row.get("Output (Final)", "")).strip()
            if not cat or not inp: continue
            if cat not in new_maps: new_maps[cat] = {}
            new_maps[cat][inp] = out
    base_rules["value_maps"] = new_maps
    return base_rules

# ==========================================================
# 3. CORE LOGIC (FULL CAPABILITY)
# ==========================================================
def smart_parse(txt: str) -> Dict[str, Any]:
    text = (txt or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"): text = text[1:]
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

def clean_input_csv(csv_text: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    parsed = smart_parse(csv_text)
    cols, rows = parsed["columns"], parsed["rows"]
    if not cols: return {"cleaned_csv": "", "diff": {}}

    rename_map, unknown, seen, new_columns = {}, [], {}, []
    col_setter_base = []

    # Map headers to standard names
    for c in cols:
        setter = get_setter(c)
        if setter:
            base = IC_TO_BASE.get(setter, setter)
            new_name = get_canonical(base, c)
            rename_map[c] = new_name
            col_setter_base.append(base)
        else:
            rename_map[c] = c
            unknown.append(c)
            col_setter_base.append(None)

    for c in cols:
        nc = rename_map[c]
        seen[nc] = seen.get(nc, 0) + 1
        new_columns.append(f"{nc} ({seen[nc]})" if seen[nc] > 1 else nc)

    # Clean cell values based on rules
    cleaned_rows = []
    value_maps = rules.get("value_maps", {})

    for r in rows:
        rr = []
        for j, v in enumerate(r):
            x = str(v or "").strip().replace(";", ",").replace("|", ",").replace("#", ",")
            x = re.sub(r"\s*,\s*", ",", x)

            if j < len(col_setter_base) and col_setter_base[j]:
                base_key = col_setter_base[j]
                # Apply map if exists
                if base_key in value_maps:
                    x = value_maps[base_key].get(x.lower(), x)
            rr.append(x)
        cleaned_rows.append(rr)

    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    w.writerow(new_columns)
    w.writerows(cleaned_rows)

    return {"cleaned_csv": buf.getvalue(), "diff": {"header_renames": rename_map, "unknown_columns": unknown}}

# --- Helpers ---
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
        return f"{float(s):.2f}ct"
    except:
        return s

def _canon_key(label: str) -> str:
    n = _norm(_strip_available(label))
    if n in ["metals", "metal"]: return "Metal"
    if n == "shape": return "Shape"
    if n in ["size", "ringsize"]: return "Ring Size"
    return _strip_available(label)

def _pretty_value(key: str, val: str) -> str:
    v = (val or "").strip()
    k = (key or "").lower()
    if "shape" in k or "metal" in k: return v.title()
    if "center" in k or "carat" in k or "ct" in k: return _fmt_ct(v)
    return v

# --- Title Logic ---
def infer_style_name_from_title(base_title: str, varying_options: List[str]) -> str:
    t = (base_title or "").strip()
    if not t: return ""
    opts = sorted({o.strip() for o in varying_options if o}, key=len, reverse=True)
    for o in opts:
        t = re.sub(rf"(?i)\b{re.escape(o)}\b", " ", t)
    t = re.sub(r"\b\d+(\.\d+)?\s*ct\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_variant_short_title(original_title: str, style_name_base: str, parts: Dict[str, str]) -> str:
    style = style_name_base or "Item"
    tokens = [v for k, v in parts.items() if v]
    return " ".join(tokens + [style]).strip()

def build_variant_description(style_name_base: str, parts: Dict[str, str], original_title: str) -> str:
    title = build_variant_short_title(original_title, style_name_base, parts)
    lines = [f"- {k}: {v}" for k, v in parts.items() if v]
    return f"{title}\n\nVariant Details:\n" + "\n".join(lines)

# --- Pricing Logic ---
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(x).replace(",", "")))
    except:
        return default

def compute_variant_price(base_price_value: Any, parts: Dict[str, str], rules: Dict[str, Any]) -> str:
    pr = rules.get("price_rules", {}) or {}
    base = _to_float(base_price_value, default=_to_float(pr.get("default_base_price", 0), 0.0))
    adjustments = pr.get("adjustments", {}) or {}

    adj_norm = {}
    for k, mapping in adjustments.items():
        adj_norm[_norm(k)] = {_norm(v): _to_float(amt) for v, amt in mapping.items()}

    price = base
    for k, v in parts.items():
        kk, vv = _norm(k), _norm(v)
        if kk in adj_norm and vv in adj_norm[kk]:
            price += adj_norm[kk][vv]
    return f"{price:.2f}"

# --- Image Logic ---
def generate_image_urls(master_stock: str, parts: Dict[str, str], rules: Dict[str, Any]) -> Dict[str, str]:
    ir = rules.get("image_rules", {}) or {}
    base_url = str(ir.get("base_url") or "")
    ext = str(ir.get("suffix") or "")
    
    # Simple token generation
    tokens = [re.sub(r"[^A-Z0-9]", "", master_stock.upper())]
    for k, v in parts.items():
        if v: tokens.append(re.sub(r"[^A-Z0-9]", "", v.upper()))
    
    filename = "_".join(tokens)
    out = {}
    for i in range(1, 5):
        out[f"Image URL {i}"] = f"{base_url}{filename}_{i}{ext}"
    return out

# --- SKU Logic ---
def shorten_sku(master_stock: str, parts: Dict[str, str], rules: Dict[str, Any]) -> str:
    sr = rules.get("sku_rules", {}) or {}
    joiner = str(sr.get("joiner") or "-")
    tokens = [master_stock.strip()]
    for k, v in parts.items():
        if v:
            # First 3 chars of value, uppercase, alphanumeric only
            t = re.sub(r"[^A-Z0-9]", "", v.upper())[:3]
            if t: tokens.append(t)
    return joiner.join(tokens)

# --- Main Expansion Function ---
def expand_inventory(csv_text: str, rules: Dict[str, Any], enable_sku: bool, enable_images: bool, enable_pricing: bool) -> Tuple[str, Dict[str, Any]]:
    data = smart_parse(csv_text)
    cols, rows = list(data["columns"]), [list(r) for r in data["rows"]]
    if not rows: return "", {"error": "No rows found."}

    # Identify Columns
    def _h(s): return _norm(s)
    master_idx = next((i for i, c in enumerate(cols) if _h(c) in ["masterstock", "master_stock"]), -1)
    if master_idx == -1: return "", {"error": "Missing 'Master stock' column."}
    
    price_idx = next((i for i, c in enumerate(cols) if _h(c) == "price"), -1)
    stock_col_name = next((c for c in cols if _h(c) in ["stocknumber", "sku"]), "Stock Number")

    # Analyze variability
    exp_meta = []
    for idx, col in enumerate(cols):
        if idx == master_idx: continue # Handle master separately
        
        # Check if column has commas in ANY row
        varies = any("," in str(r[idx]) for r in rows)
        exp_meta.append({"col": col, "idx": idx, "varies": varies, 
                         "is_available": is_available_col(col),
                         "available_base": available_base_name(col)})

    final_rows = []
    
    for i, row in enumerate(rows):
        # 1. SPLIT MASTER STOCK (Requested Feature)
        raw_master = str(row[master_idx] or "").strip()
        master_tokens = [m.strip() for m in raw_master.split(",") if m.strip()]
        if not master_tokens: master_tokens = [f"ITEM-{i+1:03}"]

        base_price_val = row[price_idx] if price_idx != -1 else 0
        orig_title = str(row[cols.index("Short Title")] if "Short Title" in cols else "")

        # Prepare tokens for this row
        row_tokens = []
        for meta in exp_meta:
            val = str(row[meta["idx"]] or "").replace("#", ",")
            ts = [t.strip() for t in val.split(",") if t.strip()] or [""]
            row_tokens.append(ts)

        # Calculate style base using ALL variations
        varying_vals = [t for idx, meta in enumerate(exp_meta) for t in row_tokens[idx] if meta["varies"]]
        
        # 2. LOOP MASTERS
        for current_master in master_tokens:
            style_base = infer_style_name_from_title(orig_title, varying_vals) or current_master

            # Cartesian Product
            for combo in product(*row_tokens):
                new_r = {}
                parts = {}
                
                # Fill data
                new_r["Master stock"] = current_master
                
                for meta, token in zip(exp_meta, combo):
                    # Handle "Available X" columns logic
                    if meta["is_available"]:
                        base = meta["available_base"]
                        new_r[f"Available {base}"] = normalize_list(str(row[meta["idx"]]))
                        new_r[base] = token
                        if meta["varies"] and token:
                            parts[_canon_key(base)] = _pretty_value(_canon_key(base), token)
                    else:
                        new_r[meta["col"]] = token
                        if meta["varies"] and token:
                            parts[_canon_key(meta["col"])] = _pretty_value(meta["col"], token)

                # SKU
                existing_sku = str(new_r.get(stock_col_name, "")).strip()
                if not existing_sku and enable_sku:
                    new_r[stock_col_name] = shorten_sku(current_master, parts, rules)
                elif not existing_sku:
                    new_r[stock_col_name] = current_master

                # Titles
                new_r["Short Title"] = build_variant_short_title(orig_title, style_base, parts)
                new_r["Description"] = build_variant_description(style_base, parts, orig_title)

                # Pricing
                if price_idx != -1 and enable_pricing:
                    new_r["Price"] = compute_variant_price(base_price_val, parts, rules)
                elif price_idx != -1:
                    new_r["Price"] = base_price_val

                # Images
                if enable_images:
                    img_map = generate_image_urls(current_master, parts, rules)
                    new_r.update(img_map)

                final_rows.append(new_r)

    # Re-order columns
    if not final_rows: return "", {"rows_out": 0}
    
    # Smart ordering
    all_keys = list(final_rows[0].keys())
    # Prioritize standard cols
    priority = ["Master stock", stock_col_name, "Short Title", "Description", "Price"]
    ordered_cols = [k for k in priority if k in all_keys] + [k for k in all_keys if k not in priority]
    
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=ordered_cols)
    w.writeheader()
    w.writerows(final_rows)
    
    return buf.getvalue(), {"rows_out": len(final_rows)}

# ==========================================================
# 4. STREAMLIT APP
# ==========================================================
def run_app():
    if "rules" not in st.session_state:
        st.session_state["rules"] = load_rules()

    # SIDEBAR
    with st.sidebar:
        st.title("Inventory Expander")
        st.markdown("Full Capability Version")
        
        st.subheader("Options")
        en_sku = st.toggle("Shorten SKUs", value=st.session_state["rules"]["sku_rules"]["enabled"])
        en_img = st.toggle("Generate Images", value=st.session_state["rules"]["image_rules"]["enabled"])
        en_prc = st.toggle("Calculate Pricing", value=st.session_state["rules"]["price_rules"].get("enabled", True))
        
        st.divider()
        st.subheader("Value Mapping Rules")
        st.info("Rename values (e.g. Y -> Yellow Gold)")
        
        # VISUAL EDITOR
        df_rules = rules_to_dataframe(st.session_state["rules"])
        edited_df = st.data_editor(
            df_rules,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Type": st.column_config.SelectboxColumn(options=["Rename"], required=True),
                "Category": st.column_config.SelectboxColumn(options=["metals", "shape", "size"], required=True)
            }
        )
        
        if st.button("Save Rules"):
            updated = dataframe_to_rules(edited_df, st.session_state["rules"])
            # Update toggles in rules as well
            updated["sku_rules"]["enabled"] = en_sku
            updated["image_rules"]["enabled"] = en_img
            save_rules(updated)
            st.session_state["rules"] = updated
            st.success("Saved!")

    # MAIN
    st.markdown("### Upload Inventory CSV")
    st.markdown("Supports: `Master Stock` comma-splitting, Price/Image rules, and Auto-cleaning.")
    
    up = st.file_uploader("Upload", type=["csv", "txt"])
    
    if up:
        raw = up.read().decode("utf-8", errors="ignore")
        
        # 1. Clean
        cleaned = clean_input_csv(raw, st.session_state["rules"])
        
        # 2. Show cleaning status
        if cleaned["diff"]["unknown_columns"]:
            st.warning(f"Unmapped Columns: {cleaned['diff']['unknown_columns']}")
        else:
            st.success("All columns mapped successfully.")
            
        # 3. Expand
        if st.button("Run Expansion", type="primary"):
            csv_out, meta = expand_inventory(
                cleaned["cleaned_csv"], 
                st.session_state["rules"], 
                enable_sku=en_sku, 
                enable_images=en_img, 
                enable_pricing=en_prc
            )
            
            if "error" in meta:
                st.error(meta["error"])
            else:
                st.success(f"Success! Generated {meta['rows_out']} variants.")
                st.download_button("Download CSV", csv_out, "expanded_inventory.csv", "text/csv")
                
                # Preview
                df = pd.read_csv(io.StringIO(csv_out))
                st.dataframe(df.head(50), use_container_width=True)

if __name__ == "__main__":
    run_app()
