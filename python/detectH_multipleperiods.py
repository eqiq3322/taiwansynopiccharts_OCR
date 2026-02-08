import cv2
import pandas as pd
import numpy as np
import easyocr
import os
import glob
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
from config import load_config_section, parse_datetime, resolve_path

# ================= 1. Parameter Settings =================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUPAN_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_IMG = "../sfcmap"
DEFAULT_DATA = "../DATA"
DEFAULT_OUTPUT = "html/detectH_multipleperiods.html"

parser = argparse.ArgumentParser(description="Detect high-pressure centers (multi-period).")
parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
parser.add_argument("--img-folder", help="Path to sfcmap folder.")
parser.add_argument("--data-folder", help="Path to DATA folder.")
parser.add_argument("--output-html", help="Output HTML path.")
args = parser.parse_args()

cfg = load_config_section(args.config, "detectH_multipleperiods")

IMG_FOLDER = resolve_path(args.img_folder or cfg.get("img_folder") or DEFAULT_IMG, PROJECT_ROOT)
DATA_FOLDER = resolve_path(args.data_folder or cfg.get("data_folder") or DEFAULT_DATA, PROJECT_ROOT)
OUTPUT_HTML = resolve_path(args.output_html or cfg.get("output_html") or DEFAULT_OUTPUT, PROJECT_ROOT)

# --- Time Settings (Multi-Period Support) ---
# Format: List of tuples (Start_Time, End_Time)
# You can add as many ranges as you need here.
DEFAULT_TIME_RANGES = [
    (datetime(2016, 12, 13, 22, 0), datetime(2016, 12, 16, 23, 0)),
    (datetime(2017, 1, 13, 0, 30), datetime(2017, 1, 16, 19, 0)),
    (datetime(2017, 1, 19, 23, 30), datetime(2017, 1, 26, 19, 0)),
    (datetime(2017, 1, 30, 8, 0), datetime(2017, 2, 2, 22, 0)),
    (datetime(2017, 2, 8, 20, 30), datetime(2017, 2, 11, 23, 30)),
    (datetime(2017, 2, 23, 10, 0), datetime(2017, 2, 27, 19, 30)),
    (datetime(2022, 10, 9, 5, 30), datetime(2022, 10, 19, 22, 0)),
    (datetime(2022, 10, 28, 8, 30), datetime(2022, 11, 2, 22, 30)),
    (datetime(2022, 11, 3, 20, 0), datetime(2022, 11, 7, 0, 0)),
    (datetime(2022, 11, 30, 1, 0), datetime(2022, 12, 15, 9, 0)),
    (datetime(2023, 1, 15, 0, 0), datetime(2023, 1, 21, 19, 0)),
    (datetime(2023, 2, 2, 1, 0), datetime(2023, 2, 6, 2, 0)),
    (datetime(2023, 2, 13, 18, 0), datetime(2023, 2, 17, 0, 0)),
]

def parse_time_ranges(raw):
    if not raw:
        return None
    ranges = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        start = parse_datetime(pair[0])
        end = parse_datetime(pair[1])
        if start and end:
            ranges.append((start, end))
    return ranges or None

TIME_RANGES_CONFIG = parse_time_ranges(cfg.get("time_ranges")) or DEFAULT_TIME_RANGES

# ROI Range (Filter range for data points)
ROI_LON_MIN, ROI_LON_MAX = 100, 150
ROI_LAT_MIN, ROI_LAT_MAX = 15, 55

# LCC Projection Parameters
LCC_M = np.array([[ 868.63533518,   1.46910273,  540.98818772], 
                  [   2.2078229 , -870.67769755, -574.29925271], 
                  [   0.        ,    0.        ,    1.        ]])
LCC_M_INV = np.linalg.inv(LCC_M)
STD_LAT1 = 30.0
STD_LAT2 = 60.0
CENTRAL_LON = 120.0

# Blue H HSV Range
BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([150, 255, 255])

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

# ================= 2. Data Processing Functions =================

def load_bsmi_data_multi_period(time_ranges):
    """
    Smart load month files based on multiple time ranges.
    """
    # 1. Identify all unique months required across all ranges
    target_months = set()
    
    for start_date, end_date in time_ranges:
        # Buffer hours for 6h average calculation
        real_start = start_date - timedelta(hours=6)
        real_end = end_date + timedelta(hours=6)
        
        curr = real_start.replace(day=1, hour=0, minute=0, second=0)
        limit = real_end.replace(day=1, hour=0, minute=0, second=0)
        
        while curr <= limit:
            target_months.add(curr.strftime("%Y%m"))
            curr += relativedelta(months=1)
            
    sorted_months = sorted(list(target_months))
    print(f"Based on multiple ranges, reading months: {sorted_months}")

    # 2. Load files
    df_list = []
    for ym in sorted_months:
        filename = f"BSMI{ym}.txt"
        file_path = DATA_FOLDER / filename
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, usecols=['TIMESTAMP', 'WS_100E', 'BP_93'], 
                                 low_memory=False, na_values=['NaN', 'NAN', '-9999'])
                df['WS_100E'] = pd.to_numeric(df['WS_100E'], errors='coerce')
                df['BP_93'] = pd.to_numeric(df['BP_93'], errors='coerce')
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                df.set_index('TIMESTAMP', inplace=True)
                df_list.append(df)
            except Exception as e:
                print(f"  ❌ Read failed {filename}: {e}")
    
    if not df_list:
        print("⚠️ Warning: No wind data found for any requested months!")
        return pd.DataFrame()

    full_df = pd.concat(df_list).sort_index()

    # 3. Filter DataFrame to include only data within the requested ranges (plus buffer)
    # We create a boolean mask for all ranges
    mask = pd.Series(False, index=full_df.index)
    
    for start_date, end_date in time_ranges:
        # Buffer for calculation
        s_buf = start_date - timedelta(hours=6)
        e_buf = end_date + timedelta(hours=6)
        mask |= ((full_df.index >= s_buf) & (full_df.index <= e_buf))
        
    final_df = full_df[mask]
    
    print(f"Data loaded! Total rows after filtering: {len(final_df)}")
    return final_df

def get_6h_avg_wind(target_dt, df):
    if df.empty: return None, None
    start_t = target_dt - timedelta(hours=3) + timedelta(seconds=1)
    end_t = target_dt + timedelta(hours=3) - timedelta(seconds=1)
    
    # Check if target time range is within df bounds roughly
    if start_t < df.index.min() or end_t > df.index.max():
        return None, None
        
    try:
        subset = df.loc[start_t:end_t]
        if subset.empty: return None, None
        return subset['WS_100E'].mean(), subset['BP_93'].mean()
    except KeyError:
        return None, None

# ================= 3. Image Processing & OCR =================

def pixel_to_latlon(px, py):
    vec = np.dot(LCC_M_INV, [px, py, 1.0])
    x = vec[0]
    y = vec[1]
    lat1_r = np.radians(STD_LAT1)
    lat2_r = np.radians(STD_LAT2)
    lon0_r = np.radians(CENTRAL_LON)
    if STD_LAT1 == STD_LAT2:
        n = np.sin(lat1_r)
    else:
        n = np.log(np.cos(lat1_r)/np.cos(lat2_r)) / np.log(np.tan(np.pi/4 + lat2_r/2)/np.tan(np.pi/4 + lat1_r/2))
    F = (np.cos(lat1_r) * np.power(np.tan(np.pi/4 + lat1_r/2), n)) / n
    rho0 = F / np.power(np.tan(np.pi/4 + np.radians(90)/2), n)
    rho = np.sqrt(x**2 + (rho0 - y)**2)
    theta = np.arctan2(x, rho0 - y)
    t = np.power(F / rho, 1/n)
    lat_r = 2 * np.arctan(t) - np.pi/2
    lon_r = theta / n + lon0_r
    return round(np.degrees(lat_r), 2), round(np.degrees(lon_r), 2)

def find_h_via_ocr(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return []
    
    print(f"--- Analysis: {Path(img_path).name} ---")

    # 1) Blue Mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    blue_pixel_count = cv2.countNonZero(mask_blue)
    
    if blue_pixel_count < 500:
        print("  -> [Log] Junk map! (No blue high pressure area)")
        return []

    # 2) Dilate
    kernel_big = np.ones((15, 15), np.uint8)
    mask_dil = cv2.dilate(mask_blue, kernel_big, iterations=1)
    
    # 3) Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dil, connectivity=8)
    
    valid_h_list = []
    h_img, w_img = img.shape[:2]
    
    for label_id in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label_id]
        
        if area < 200 or w_box > 200 or h_box > 200: continue
        if w_box < 20 or h_box < 20: continue

        pad = 5
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w_img, x + w_box + pad); y1 = min(h_img, y + h_box + pad)
        
        sub_img = img[y0:y1, x0:x1].copy()
        sub_mask = mask_blue[y0:y1, x0:x1]
        
        white_bg = np.full_like(sub_img, 255)
        blue_only = np.where(sub_mask[..., None] > 0, sub_img, white_bg)
        
        if cv2.countNonZero(sub_mask) < 30: continue
        
        try:
            ocr_texts = reader.readtext(blue_only, allowlist='0123456789Hh', detail=0)
        except Exception as e:
            continue
            
        if not ocr_texts: continue
        joined = ''.join(ocr_texts).replace(' ', '').upper()
        
        if 'H' not in joined: continue
        
        digits = ''.join(ch for ch in joined if ch.isdigit())
        if 3 <= len(digits) <= 4:
            val = int(digits)
            if 900 < val < 1100:
                cx = x + w_box // 2
                cy = y + h_box // 2
                lat, lon = pixel_to_latlon(cx, cy)
                
                if ROI_LAT_MIN <= lat <= ROI_LAT_MAX and ROI_LON_MIN <= lon <= ROI_LON_MAX:
                    print(f"    -> [Locked CC] Blue H: ({lat}, {lon})  P={val}")
                    valid_h_list.append({
                        'Lat': lat, 'Lon': lon, 'Pressure_Center': val, 'Method': 'Blue_H_Blob'
                    })

    return valid_h_list

# ================= 4. Main Program =================

if __name__ == "__main__":
    print("=== Start Processing (Multi-Period Mode) ===")
    
    # Load data for all requested periods
    df_wind = load_bsmi_data_multi_period(TIME_RANGES_CONFIG)
    
    final_data = []

    # Iterate through each defined time range
    for start_t, end_t in TIME_RANGES_CONFIG:
        print(f"\nProcessing Range: {start_t} to {end_t}")
        current_time = start_t
        
        while current_time <= end_t:
            time_str = current_time.strftime("%Y%m%d_%H00")
            img_path_gif = IMG_FOLDER / f"sfcmap_{time_str}.gif"
            img_path_png = IMG_FOLDER / f"sfcmap_{time_str}.png"
            
            target_img = None
            if os.path.exists(img_path_gif): target_img = img_path_gif
            elif os.path.exists(img_path_png): target_img = img_path_png
            
            if target_img:
                ws, bp = get_6h_avg_wind(current_time, df_wind)
                
                # Filter Condition: Skip if ws < 10 or no data
                if ws is not None:
                    if ws < 10:
                        # Log it or just silently skip
                        # print(f"  -> Skipping {time_str}, WindSpeed {ws:.2f} < 10")
                        pass 
                    else:
                        # Wind Speed is valid (>= 10), proceed to OCR
                        h_data_list = find_h_via_ocr(target_img)
                        for item in h_data_list:
                            final_data.append({
                                "Time": current_time.strftime("%Y-%m-%d %H:%M"),
                                "Lat": item['Lat'], "Lon": item['Lon'],
                                "WindSpeed": round(ws, 2),
                                "Pressure_Tower": round(bp, 1) if bp else "N/A",
                                "Pressure_Center": item['Pressure_Center'],
                                #"Detect_Method": item['Method']
                            })
                            
            current_time += timedelta(hours=6)

    # Plotting
    if final_data:
        df_res = pd.DataFrame(final_data)
        df_res['Pressure_Center'] = df_res['Pressure_Center'].astype(str)

        print(f"\nCollected {len(df_res)} valid data points, drawing map...")

        # --- Prepare Date Period Annotation Text ---
        period_texts = ["<b>Plotting period:</b>"] # Bold heading
        for start_t, end_t in TIME_RANGES_CONFIG:
            s_str = start_t.strftime("%Y/%m/%d")
            e_str = end_t.strftime("%Y/%m/%d")
            if s_str == e_str:
                period_texts.append(s_str)
            else:
                period_texts.append(f"{s_str} - {e_str}")
        # Join with HTML line breaks for Plotly annotation
        annotation_text = "<br>".join(period_texts)

        # --- Updated Title ---
        chart_title = "6hr High Pressure Centers and BSMI WS"

        # Create Geo Scatter Plot
        fig = px.scatter_geo(
            df_res,
            lat='Lat', lon='Lon', color='WindSpeed',
            # Modified Range and Color Scale as requested
            range_color=[10, 30], 
            color_continuous_scale='Turbo', # Using 'Turbo' for distinct look (Blue->Red but different saturation)
            hover_name='Time',
            hover_data={
                'Lat': ':.2f', 'Lon': ':.2f', 'WindSpeed': ':.2f m/s', 
                'Pressure_Tower': True, 'Pressure_Center': True, 'Detect_Method': True
            },
            title=chart_title
        )
        
        fig.update_layout(title_x=0.5)
        fig.update_coloraxes(colorbar_title="Wind Speed (m/s)")

        # Configure Geo Layout
        fig.update_geos(
            projection_type="conic conformal",
            projection_parallels=[30, 60],
            projection_rotation=dict(lon=120),
            
            lataxis_range=[5, 65], 
            lonaxis_range=[95, 155],
            
            showcountries=True, showcoastlines=True, 
            coastlinecolor="Black", countrycolor="Black",
            
            showframe=False,
            
            lonaxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.5, dtick=5),
            lataxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.5, dtick=5)
        )

        # === Add Manual Black Frame (100-150E, 10-60N) ===
        lons_bottom = list(range(100, 151, 1))
        lats_bottom = [10] * len(lons_bottom)
        
        lats_right = list(range(10, 61, 1))
        lons_right = [150] * len(lats_right)
        
        lons_top = list(range(150, 99, -1))
        lats_top = [60] * len(lons_top)
        
        lats_left = list(range(60, 9, -1))
        lons_left = [100] * len(lats_left)
        
        frame_lons = lons_bottom + lons_right + lons_top + lons_left
        frame_lats = lats_bottom + lats_right + lats_top + lats_left
        
        fig.add_trace(go.Scattergeo(
            mode="lines",
            lon=frame_lons,
            lat=frame_lats,
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip"
        ))

        # === Add Manual Labels ===
        
        # 1. Longitude Labels
        lon_labels = list(range(100, 151, 10)) 
        fig.add_trace(go.Scattergeo(
            mode="text",
            lon=lon_labels,
            lat=[10] * len(lon_labels),
            text=[f"{l}E" for l in lon_labels],
            textposition="bottom center",
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # 2. Latitude Labels (Left)
        lat_labels = list(range(10, 61, 10))
        fig.add_trace(go.Scattergeo(
            mode="text",
            lon=[100] * len(lat_labels),
            lat=lat_labels,
            text=[f"{l}N" for l in lat_labels],
            textposition="middle left",
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # 3. Latitude Labels (Right)
        fig.add_trace(go.Scattergeo(
            mode="text",
            lon=[150] * len(lat_labels),
            lat=lat_labels,
            text=[f"{l}N" for l in lat_labels],
            textposition="middle right",
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # === Add Left Margin Annotation for Periods ===
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=-0.02, y=1.0, # <--- 【修改這裡 1】: 原本是 -0.15，改成 -0.02 (往右移)
            showarrow=False,
            xanchor="left", yanchor="top",
            align="left",
            font=dict(size=11, color="black"),
            bordercolor="black", borderwidth=1, borderpad=4, 
            bgcolor="white" 
        )

        # Adjust layout margins - Increased left margin for the annotation
        fig.update_layout(
            height=900, 
            margin={"r": 80, "t": 80, "l": 120, "b": 80} # <--- 【修改這裡 2】: 原本是 150，改成 120 (稍微縮小左邊界)
        )
        
        OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUT_HTML))
        print(f"✅ Finished! Result saved to: {OUTPUT_HTML}")
    else:
        print("\n⚠️ Execution finished, no valid data found (possibly all WS < 10 or missing data).")
