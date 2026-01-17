# ============================
# Perfect Love → Shopify Products with Metafields
# ============================
# HOW TO USE (in Colab):
# 1. Run this cell.
# 2. Use the file picker to upload your "Combined" inventory CSV.
# 3. Script auto-detects the uploaded filename, processes it, and
#    creates a Shopify-ready CSV with products, variants, tags, and metafields.
# 4. A download link for the output CSV will appear at the end.

import re
import numpy as np
import pandas as pd

# -------- Colab UI imports --------
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==============
# Helper: slugify for Shopify handles
# ==============
def slugify(text):
    if pd.isna(text) or text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

# ==============
# Step 1: Upload CSV via UI
# ==============
if IN_COLAB:
    print("📂 Please upload your 'Combined Inventory' CSV (e.g. Perfect Love Inventory - Combined Sheet - Sheet7 (1).csv)")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file uploaded. Please run the cell again and upload a CSV.")

    # Take the first uploaded file
    INVENTORY_CSV = list(uploaded.keys())[0]
    print(f"✅ Using uploaded file: {INVENTORY_CSV}")
else:
    # Fallback for local debugging (edit the filename as needed)
    INVENTORY_CSV = "Perfect Love Inventory - Combined Sheet - Sheet7 (1).csv"
    print(f"Not in Colab, using local file: {INVENTORY_CSV}")

# Output file name (based on input name)
OUTPUT_CSV = INVENTORY_CSV.rsplit(".", 1)[0] + "_SHOPIFY_WITH_METAFIELDS.csv"

# ==============
# Load combined inventory
# ==============
df = pd.read_csv(INVENTORY_CSV)

print("\nColumns in inventory CSV:")
print(df.columns.tolist())

# ==============
# Shopify base columns (same schema as your working file)
# ==============
shopify_cols = [
    "Handle",
    "Title",
    "Body (HTML)",
    "Vendor",
    "Product Category",
    "Type",
    "Tags",
    "Published",
    "Option1 Name",
    "Option1 Value",
    "Option1 Linked To",
    "Option2 Name",
    "Option2 Value",
    "Option2 Linked To",
    "Option3 Name",
    "Option3 Value",
    "Option3 Linked To",
    "Variant SKU",
    "Variant Grams",
    "Variant Inventory Tracker",
    "Variant Inventory Qty",
    "Variant Inventory Policy",
    "Variant Fulfillment Service",
    "Variant Price",
    "Variant Compare At Price",
    "Variant Requires Shipping",
    "Variant Taxable",
    "Unit Price Total Measure",
    "Unit Price Total Measure Unit",
    "Unit Price Base Measure",
    "Unit Price Base Measure Unit",
    "Variant Barcode",
    "Image Src",
    "Image Position",
    "Image Alt Text",
    "Gift Card",
    "SEO Title",
    "SEO Description",
    "Style (product.metafields.custom.style)",
    "Complementary products (product.metafields.shopify--discovery--product_recommendation.complementary_products)",
    "Related products (product.metafields.shopify--discovery--product_recommendation.related_products)",
    "Related products settings (product.metafields.shopify--discovery--product_recommendation.related_products_display)",
    "Search product boosts (product.metafields.shopify--discovery--product_search_boost.queries)",
    "Variant Image",
    "Variant Weight Unit",
    "Variant Tax Code",
    "Cost per item",
    "Status",
]

# ==============
# Define metafield columns
# ==============
meta_base = "metafields_global_namespace_key[single_line_text].vdbjl."

metafield_cols = [
    meta_base + "vdb_stock_id",             # left blank (no stock_id in source)
    meta_base + "vdb_stock_num",
    meta_base + "type",
    meta_base + "metal",
    meta_base + "item_location",
    meta_base + "side_stone_color",
    meta_base + "side_stone_clarity",
    # extra details into metafields:
    meta_base + "jewelry_classification",
    meta_base + "shape",
    meta_base + "weight",
    meta_base + "available_diamond_spread",
    meta_base + "available_metal_type",
    meta_base + "available_shape",
    meta_base + "customizable",
]

all_cols = shopify_cols + metafield_cols
rows_out = []

# ==============
# Tags helper — keep original + add extras + ensure 'vdbjl'
# ==============
def build_tags_from_master(master_row):
    # Prefer Tags.1 if present, else Tags
    orig_tags = None
    if "Tags.1" in master_row and isinstance(master_row["Tags.1"], str):
        orig_tags = master_row["Tags.1"]
    elif "Tags" in master_row and isinstance(master_row["Tags"], str):
        orig_tags = master_row["Tags"]
    else:
        orig_tags = ""

    orig_tags = orig_tags.strip()
    base_list = orig_tags.split(",") if orig_tags else []
    base_list = [t for t in base_list if t != ""]  # keep as-is, just drop empties

    first_tag = base_list[0].strip() if base_list else ""
    existing = {t.strip() for t in base_list}

    extras = []

    # Product Family_<first tag with spaces as _>
    if first_tag:
        pf_tag = f"Product Family_{first_tag.replace(' ', '_')}"
        if pf_tag not in existing:
            extras.append(pf_tag)

    # Item Location_United States
    il_tag = "Item Location_United States"
    if il_tag not in existing:
        extras.append(il_tag)

    # vdb_stock_num_<master Stock Number>
    master_stock_num = master_row.get("Stock Number")
    if isinstance(master_stock_num, str) and master_stock_num:
        vdb_tag = f"vdb_stock_num_{master_stock_num}"
        if vdb_tag not in existing:
            extras.append(vdb_tag)

    combined = base_list + extras

    # Ensure 'vdbjl' is present
    if "vdbjl" not in {t.strip() for t in combined}:
        combined.append("vdbjl")

    return ",".join(combined) if combined else np.nan

# ==============
# Main transform: one product per Master Stock Number
# ==============
for msn, group in df.groupby("Master Stock Number"):
    group = group.copy()

    # Put master row first (is_master_product=True) if exists
    if "is_master_product" in group.columns and group["is_master_product"].any():
        group = group.sort_values("is_master_product", ascending=False).reset_index(drop=True)
    else:
        group = group.reset_index(drop=True)

    master = group.iloc[0]

    # Handle & title
    short_title = master.get("Short Title")
    stock_number_master = master.get("Stock Number")

    base_for_handle = short_title if isinstance(short_title, str) and short_title else stock_number_master
    handle = slugify(base_for_handle)

    title = short_title if isinstance(short_title, str) and short_title else stock_number_master

    # Description → Body (HTML)
    description = master.get("Description")
    body_html = f"<p>{description}</p>" if isinstance(description, str) and description else np.nan

    vendor = "Perfect Love Inventory"
    prod_type = "Jewelry"

    # Build tags for this product
    tags_str = build_tags_from_master(master)

    # ==========
    # Build rows for each variant
    # ==========
    for idx, (_, row) in enumerate(group.iterrows()):
        out = {c: np.nan for c in all_cols}
        out["Handle"] = handle

        # ----- Product header row -----
        if idx == 0:
            out["Title"] = title
            out["Body (HTML)"] = body_html
            out["Vendor"] = vendor
            out["Product Category"] = np.nan
            out["Type"] = prod_type
            out["Tags"] = tags_str
            out["Published"] = True
            out["Option1 Name"] = "Metal Type"
            out["Option2 Name"] = "Available Diamond Spread"
            out["Gift Card"] = False
            out["Status"] = "active"
        else:
            # Non-header rows: variant-level only
            out["Title"] = np.nan
            out["Body (HTML)"] = np.nan
            out["Published"] = np.nan
            out["Status"] = np.nan
            # Option1/2 Name left blank on non-header rows

        # Clear Option3
        out["Option3 Name"] = np.nan
        out["Option3 Value"] = np.nan
        out["Option3 Linked To"] = np.nan

        # Variant options
        out["Option1 Value"] = row.get("Metal")
        out["Option2 Value"] = row.get("Diamond Spread")

        # Variant ID & inventory
        out["Variant SKU"] = row.get("Stock Number")
        out["Variant Grams"] = 0
        out["Variant Inventory Tracker"] = "shopify"
        out["Variant Inventory Qty"] = 1
        out["Variant Inventory Policy"] = "deny"
        out["Variant Fulfillment Service"] = "manual"

        price = row.get("Price")
        out["Variant Price"] = price
        out["Cost per item"] = price
        out["Variant Compare At Price"] = np.nan

        out["Variant Requires Shipping"] = True
        out["Variant Taxable"] = True
        out["Variant Weight Unit"] = "lb"

        # Images
        img = row.get("Image URL 1")
        if isinstance(img, str) and img:
            out["Image Src"] = img
            out["Variant Image"] = img
            out["Image Position"] = idx + 1

            metal_slug = str(row.get("Metal") or "").strip().lower().replace(" ", "-")
            spread = str(row.get("Diamond Spread") or "").strip()
            alt_parts = []
            if metal_slug:
                alt_parts.append(metal_slug)
            if spread:
                alt_parts.append(spread)
            if alt_parts:
                out["Image Alt Text"] = "-".join(alt_parts)

        # ----- Metafields only on header row -----
        if idx == 0:
            # vdb_stock_id intentionally left blank
            out[meta_base + "vdb_stock_id"] = np.nan

            out[meta_base + "vdb_stock_num"] = stock_number_master
            out[meta_base + "type"] = master.get("Jewelry Type")
            out[meta_base + "metal"] = master.get("Metal")
            out[meta_base + "item_location"] = "United States"
            out[meta_base + "side_stone_color"] = master.get("Side Color")
            out[meta_base + "side_stone_clarity"] = master.get("Side Clarity")

            # Extra details as metafields
            out[meta_base + "jewelry_classification"] = master.get("Jewelry Classification")
            out[meta_base + "shape"] = master.get("Shape")
            out[meta_base + "weight"] = master.get("Weight")
            out[meta_base + "available_diamond_spread"] = master.get("Available Diamond Spread")
            out[meta_base + "available_metal_type"] = master.get("Available Metal Type")
            out[meta_base + "available_shape"] = master.get("Available Shape")
            out[meta_base + "customizable"] = master.get("Customizable")

        rows_out.append(out)

# ==============
# Build final DataFrame & save
# ==============
out_df = pd.DataFrame(rows_out, columns=all_cols)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Done! Wrote {len(out_df)} rows to: {OUTPUT_CSV}")

# ==============
# Step 3: Download link in Colab
# ==============
if IN_COLAB:
    print("⬇️ Click below to download the Shopify CSV:")
    files.download(OUTPUT_CSV)
else:
    print("Not in Colab, file saved locally.")
