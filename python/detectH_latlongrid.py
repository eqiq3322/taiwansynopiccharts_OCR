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
DEFAULT_OUTPUT = "html/detectH_latlongrid.html"
DEFAULT_START = "2024-10-01 00:00"
DEFAULT_END = "2025-03-30 23:59"

parser = argparse.ArgumentParser(description="Detect high-pressure centers with lat/lon frame.")
parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
parser.add_argument("--img-folder", help="Path to sfcmap folder.")
parser.add_argument("--data-folder", help="Path to DATA folder.")
parser.add_argument("--output-html", help="Output HTML path.")
parser.add_argument("--start", help="Start time (YYYY-MM-DD HH:MM).")
parser.add_argument("--end", help="End time (YYYY-MM-DD HH:MM).")
args = parser.parse_args()

cfg = load_config_section(args.config, "detectH_latlongrid")

IMG_FOLDER = resolve_path(args.img_folder or cfg.get("img_folder") or DEFAULT_IMG, PROJECT_ROOT)
DATA_FOLDER = resolve_path(args.data_folder or cfg.get("data_folder") or DEFAULT_DATA, PROJECT_ROOT)
OUTPUT_HTML = resolve_path(args.output_html or cfg.get("output_html") or DEFAULT_OUTPUT, PROJECT_ROOT)

START_TIME = parse_datetime(args.start or cfg.get("start") or DEFAULT_START)
END_TIME = parse_datetime(args.end or cfg.get("end") or DEFAULT_END)

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

def load_bsmi_data_smart(start_date, end_date):
    """Smart load month files."""
    real_start = start_date - timedelta(hours=6)
    real_end = end_date + timedelta(hours=6)
    
    target_months = []
    curr = real_start.replace(day=1, hour=0, minute=0, second=0)
    limit = real_end.replace(day=1, hour=0, minute=0, second=0)
    
    while curr <= limit:
        target_months.append(curr.strftime("%Y%m"))
        curr += relativedelta(months=1)
        
    print(f"Based on settings, reading months: {target_months}")

    df_list = []
    for ym in target_months:
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
    
    if df_list:
        full_df = pd.concat(df_list).sort_index()
        cut_start = start_date - timedelta(hours=6)
        cut_end = end_date + timedelta(hours=6)
        final_df = full_df.loc[cut_start:cut_end]
        print(f"Data loaded! Range: {final_df.index.min()} to {final_df.index.max()}")
        return final_df
    else:
        print("⚠️ Warning: No wind data found!")
        return pd.DataFrame()

def get_6h_avg_wind(target_dt, df):
    if df.empty: return None, None
    start_t = target_dt - timedelta(hours=3) + timedelta(seconds=1)
    end_t = target_dt + timedelta(hours=3) - timedelta(seconds=1)
    if start_t < df.index.min() or end_t > df.index.max():
        return None, None
    subset = df.loc[start_t:end_t]
    if subset.empty: return None, None
    return subset['WS_100E'].mean(), subset['BP_93'].mean()

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

    # 2) Dilate (Merge H and Number)
    kernel_big = np.ones((15, 15), np.uint8)
    mask_dil = cv2.dilate(mask_blue, kernel_big, iterations=1)
    
    # 3) Connected Components (Blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dil, connectivity=8)
    
    valid_h_list = []
    h_img, w_img = img.shape[:2]
    
    for label_id in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label_id]
        
        # Filter noise or too large areas
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
    print("=== Start Processing (Manual Frame & Outer Labels) ===")
    
    df_wind = load_bsmi_data_smart(START_TIME, END_TIME)
    
    final_data = []
    current_time = START_TIME
    
    if not df_wind.empty:
        while current_time <= END_TIME:
            time_str = current_time.strftime("%Y%m%d_%H00")
            img_path_gif = IMG_FOLDER / f"sfcmap_{time_str}.gif"
            img_path_png = IMG_FOLDER / f"sfcmap_{time_str}.png"
            
            target_img = None
            if os.path.exists(img_path_gif): target_img = img_path_gif
            elif os.path.exists(img_path_png): target_img = img_path_png
            
            if target_img:
                ws, bp = get_6h_avg_wind(current_time, df_wind)
                if ws is not None:
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

        # Dynamic Title
        date_range_str = f"{START_TIME.strftime('%Y/%m/%d')} - {END_TIME.strftime('%Y/%m/%d')}"
        chart_title = f"6hr High Pressure Centers and BSMI WS during {date_range_str}"

        # Create Geo Scatter Plot
        fig = px.scatter_geo(
            df_res,
            lat='Lat', lon='Lon', color='WindSpeed',
            range_color=[0, 30], hover_name='Time',
            hover_data={
                'Lat': ':.2f', 'Lon': ':.2f', 'WindSpeed': ':.2f m/s', 
                'Pressure_Tower': True, 'Pressure_Center': True, 'Detect_Method': True
            },
            color_continuous_scale='Jet',
            title=chart_title
        )
        
        fig.update_layout(title_x=0.5)
        fig.update_coloraxes(colorbar_title="Wind Speed (m/s)")

        # Configure Geo Layout
        # STRATEGY: 
        # 1. Expand lonaxis/lataxis range slightly (5-65N, 95-155E) so labels outside the 10-60/100-150 box are visible.
        # 2. Disable default 'showframe' (because it would frame the expanded 5-65/95-155 area).
        # 3. Enable grid (showgrid=True). The grid lines will extend from the inner box to the edge of the expanded view.
        fig.update_geos(
            projection_type="conic conformal",
            projection_parallels=[30, 60],
            projection_rotation=dict(lon=120),
            
            # Expanded View Range (padding for labels)
            lataxis_range=[5, 65],  # 5 degrees padding top/bottom
            lonaxis_range=[95, 155], # 5 degrees padding left/right
            
            showcountries=True, showcoastlines=True, 
            coastlinecolor="Black", countrycolor="Black",
            
            showframe=False, # Disable default frame
            
            # Grid lines
            lonaxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.5, dtick=5),
            lataxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.5, dtick=5)
        )

        # === Add Manual Black Frame (100-150E, 10-60N) ===
        # We need to densify points along the latitude lines (Top/Bottom) 
        # so they curve correctly in the Conic projection.
        
        # Bottom Edge (10N): 100E -> 150E
        lons_bottom = list(range(100, 151, 1))
        lats_bottom = [10] * len(lons_bottom)
        
        # Right Edge (150E): 10N -> 60N (Meridians are usually straight/simple, fewer points ok, but consisten is good)
        lats_right = list(range(10, 61, 1))
        lons_right = [150] * len(lats_right)
        
        # Top Edge (60N): 150E -> 100E
        lons_top = list(range(150, 99, -1))
        lats_top = [60] * len(lons_top)
        
        # Left Edge (100E): 60N -> 10N
        lats_left = list(range(60, 9, -1))
        lons_left = [100] * len(lats_left)
        
        # Combine into one closed path
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

        # === Add Manual Labels (OUTSIDE THE FRAME) ===
        # Note: cliponaxis removed. Text will show because view range is expanded.
        
        # 1. Longitude Labels (Bottom edge at 10N)
        lon_labels = list(range(100, 151, 10)) 
        fig.add_trace(go.Scattergeo(
            mode="text",
            lon=lon_labels,
            lat=[10] * len(lon_labels),
            text=[f"{l}E" for l in lon_labels],
            textposition="bottom center", # Pushes text into the padding area (5N-10N)
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # 2. Latitude Labels (Left edge at 100E)
        lat_labels = list(range(10, 61, 10))
        fig.add_trace(go.Scattergeo(
            mode="text",
            lon=[100] * len(lat_labels),
            lat=lat_labels,
            text=[f"{l}N" for l in lat_labels],
            textposition="middle left", # Pushes text into the padding area (95E-100E)
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # 3. Latitude Labels (Right edge at 150E)
        fig.add_trace(go.Scattergeo(
            mode="text",
            lon=[150] * len(lat_labels),
            lat=lat_labels,
            text=[f"{l}N" for l in lat_labels],
            textposition="middle right", # Pushes text into the padding area (150E-155E)
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # Adjust layout margins
        fig.update_layout(
            height=900, 
            margin={"r": 80, "t": 80, "l": 80, "b": 80}
        )
        
        OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUT_HTML))
        print(f"✅ Finished! Result saved to: {OUTPUT_HTML}")
    else:
        print("\n⚠️ Execution finished, no valid data found.")
