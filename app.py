%%writefile app.py
import os, csv, io, json, re
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
# 0. CONFIG & SETUP
# ==========================================================
st.set_page_config(page_title="Inventory Expander", layout="wide")

def get_ist_now():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

MAPPING_FILE = "data-headers.csv"
RULES_FILE = "normalization_rules.json"

# ==========================================================
# 1. RULES & DATA IO (UPDATED FOR VISUAL EDITOR)
# ==========================================================
def ensure_files_exist():
    # 1. Rules
    if not os.path.exists(RULES_FILE):
        base_rules = {
            "version": 1,
            "value_maps": {
                "metals": {"y": "Yellow Gold", "w": "White Gold", "r": "Rose Gold"},
                "shape": {"rnd": "Round", "ov": "Oval", "em": "Emerald"}
            },
            "sku_rules": {"enabled": True},
            "image_rules": {"enabled": False},
            "price_rules": {"enabled": True}
        }
        with open(RULES_FILE, "w") as f:
            json.dump(base_rules, f, indent=2)

    # 2. Header Mappings
    if not os.path.exists(MAPPING_FILE):
        df = pd.DataFrame([
            {"Setter": "stock_num", "Variations": "Stock Number,SKU,Style#"},
            {"Setter": "price", "Variations": "Price,MSRP,Retail Price"},
            {"Setter": "master_stock", "Variations": "Master Stock,Group ID,Parent SKU"},
        ])
        df.to_csv(MAPPING_FILE, index=False)

def load_rules():
    ensure_files_exist()
    with open(RULES_FILE, "r") as f:
        return json.load(f)

def save_rules(rules):
    rules["updated_at"] = get_ist_now().isoformat()
    with open(RULES_FILE, "w") as f:
        json.dump(rules, f, indent=2)

# --- VISUAL EDITOR HELPERS ---
def rules_to_dataframe(rules):
    """Converts nested JSON rules into a flat table for the UI."""
    rows = []
    # 1. Value Maps (Renaming)
    maps = rules.get("value_maps", {})
    for category, mappings in maps.items():
        for k, v in mappings.items():
            rows.append({"Type": "Rename", "Category": category, "Input (CSV)": k, "Output (Final)": v})
    return pd.DataFrame(rows)

def dataframe_to_rules(df, base_rules):
    """Converts the UI table back into nested JSON."""
    new_maps = {}
    
    # Filter for Rename rows
    rename_df = df[df["Type"] == "Rename"]
    
    for _, row in rename_df.iterrows():
        cat = str(row["Category"]).strip().lower()
        inp = str(row["Input (CSV)"]).strip().lower()
        out = str(row["Output (Final)"]).strip()
        
        if not cat or not inp: continue
        
        if cat not in new_maps: new_maps[cat] = {}
        new_maps[cat][inp] = out
    
    base_rules["value_maps"] = new_maps
    return base_rules

# ==========================================================
# 2. CORE LOGIC (CSV PROCESSING)
# ==========================================================
def smart_parse(txt):
    f = io.StringIO(txt.strip())
    try: dialect = csv.Sniffer().sniff(txt[:1000], delimiters=",\t;")
    except: dialect = None
    reader = csv.reader(f, dialect=dialect) if dialect else csv.reader(f)
    rows = list(reader)
    if not rows: return [], []
    return [c.strip() for c in rows[0]], rows[1:]

def clean_csv(raw_csv, rules):
    cols, rows = smart_parse(raw_csv)
    if not cols: return None
    
    # Header Mapping Logic
    setter_map = {} # {col_index: "metals"}
    
    # Load Header Map
    map_df = pd.read_csv(MAPPING_FILE)
    # Simple normalizer
    def _n(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

    # Build lookup dict
    var_lookup = {}
    for _, r in map_df.iterrows():
        setter = r["Setter"]
        vars_ = [v.strip() for v in str(r["Variations"]).split(",") if v.strip()]
        for v in vars_: var_lookup[_n(v)] = setter
    
    # Standard Categories (Hardcoded helpers)
    cat_keywords = {
        "metals": ["metal", "goldtype", "material"],
        "shape": ["shape", "cut"],
        "size": ["size", "ringsize"],
    }

    cleaned_rows = []
    value_maps = rules.get("value_maps", {})

    for r in rows:
        new_r = []
        for i, cell in enumerate(r):
            val = cell.strip()
            # Try to map value if column header matches a category
            col_name = _n(cols[i])
            
            # Find category
            active_cat = None
            for cat, keywords in cat_keywords.items():
                if any(k in col_name for k in keywords):
                    active_cat = cat
                    break
            
            # Apply Replacement
            if active_cat and active_cat in value_maps:
                # Check exact match or lowercase match
                val = value_maps[active_cat].get(val.lower(), val)
            
            new_r.append(val)
        cleaned_rows.append(new_r)
        
    return cols, cleaned_rows

def expand_logic(cols, rows, rules, opts):
    # Find Master Stock
    def _n(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
    master_idx = next((i for i, c in enumerate(cols) if _n(c) in ["masterstock", "master_stock"]), -1)
    
    if master_idx == -1:
        return None, "❌ Error: Could not find 'Master Stock' column."

    output = []
    
    for row_idx, r in enumerate(rows):
        # 1. Handle Comma-Separated Master Stock
        raw_masters = r[master_idx].replace(";", ",").split(",")
        masters = [m.strip() for m in raw_masters if m.strip()]
        if not masters: masters = [f"ITEM-{row_idx}"]

        # 2. Parse attributes
        attrs = []
        for i, cell in enumerate(r):
            if i == master_idx: 
                attrs.append({"val": [""], "name": cols[i]}) # Placeholder
                continue
            
            # Split cell by comma
            tokens = [t.strip() for t in cell.split(",") if t.strip()]
            if not tokens: tokens = [""]
            attrs.append({"val": tokens, "name": cols[i]})

        # 3. Generate Rows
        for m_stock in masters:
            # Create style name base
            style_base = m_stock 
            
            # Cartesian Product
            # We filter out the master column from the product inputs
            prod_inputs = [x["val"] for i, x in enumerate(attrs) if i != master_idx]
            
            for combo in product(*prod_inputs):
                new_row = {}
                new_row["Master Stock"] = m_stock
                
                # Reconstruct row
                combo_iter = iter(combo)
                sku_parts = []
                
                for i, col_name in enumerate(cols):
                    if i == master_idx: continue
                    
                    val = next(combo_iter)
                    new_row[col_name] = val
                    
                    if val and len(attrs[i]["val"]) > 1: # Only add to SKU if it was a variation
                        sku_parts.append(val[:3].upper())

                # SKU Generation
                if opts["sku"]:
                    suffix = "-".join(sku_parts)
                    sku = f"{m_stock}-{suffix}" if suffix else m_stock
                    new_row["Generated SKU"] = sku

                output.append(new_row)

    return pd.DataFrame(output), None

# ==========================================================
# 3. UI LAYOUT
# ==========================================================
# Load Rules
if "rules" not in st.session_state:
    st.session_state["rules"] = load_rules()

# --- SIDEBAR (THE VISUAL EDITOR) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Run Options")
    opt_sku = st.toggle("Generate SKUs", value=True)
    opt_img = st.toggle("Generate Images", value=False)
    
    st.divider()
    st.subheader("📝 Text Replacement Rules")
    st.info("Edit this table to auto-fix values (e.g. 'Y' -> 'Yellow Gold').")

    # Convert JSON to Table
    df_rules = rules_to_dataframe(st.session_state["rules"])
    
    # RENDER EDITOR
    edited_df = st.data_editor(
        df_rules, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(options=["Rename"], required=True),
            "Category": st.column_config.SelectboxColumn(options=["metals", "shape", "size"], required=True),
        }
    )

    if st.button("💾 Save Rules"):
        # Convert Table back to JSON
        updated_rules = dataframe_to_rules(edited_df, st.session_state["rules"])
        save_rules(updated_rules)
        st.session_state["rules"] = updated_rules
        st.success("Rules saved!")

# --- MAIN PAGE ---
st.title("Inventory Expander Tool")
st.markdown("Upload a CSV file. If the `Master Stock` contains commas (e.g. `StyleA, StyleB`), they will be split into separate rows automatically.")

up = st.file_uploader("Upload CSV", type=["csv", "txt"])

if up:
    raw = up.read().decode("utf-8", errors="ignore")
    cols, cleaned_rows = clean_csv(raw, st.session_state["rules"])
    
    if cols:
        st.success(f"Loaded {len(cleaned_rows)} rows. Columns: {', '.join(cols)}")
        
        if st.button("🚀 Run Expansion", type="primary"):
            df_result, err = expand_logic(cols, cleaned_rows, st.session_state["rules"], {"sku": opt_sku})
            
            if err:
                st.error(err)
            else:
                st.subheader("Result Preview")
                st.dataframe(df_result.head(50), use_container_width=True)
                
                # CSV Download
                csv_buffer = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Expanded CSV",
                    data=csv_buffer,
                    file_name="expanded_inventory.csv",
                    mime="text/csv"
                )