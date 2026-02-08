import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import re

# ================= 設定區 =================
# 資料路徑 (根據您的描述設定)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUPAN_ROOT = PROJECT_ROOT.parents[1]
DATA_PATH = CUPAN_ROOT / "DATA"

# 定義三個塔的經緯度 (若後續要畫圖用)
TOWER_LOCS = {
    'TGC': {'lat': 24.048833, 'lon': 120.277139},
    'BSMI': {'lat': 24.312639, 'lon': 120.526750},
    'TP': {'lat': 24.000889, 'lon': 120.272944}
}

# ================= 核心邏輯函式 =================

def parse_best_sensors(columns):
    """
    解析 Header，找出 WS (風速) 和 WD (風向) 最高高度的欄位名稱。
    規則：
    1. 找 WS_, WD_ 開頭
    2. 忽略數字後的英文字 (如 95B -> 95)
    3. 取高度最高者
    4. 若高度相同，取列表中的第一個
    """
    
    def get_height(col_name):
        # 使用正則表達式提取數字部分，例如 WS_100E -> 100
        match = re.search(r'_(\d+)', col_name)
        if match:
            return int(match.group(1))
        return 0

    ws_cols = [c for c in columns if c.startswith('WS_')]
    wd_cols = [c for c in columns if c.startswith('WD_')]
    
    # 根據高度排序 (由大到小)
    ws_cols.sort(key=get_height, reverse=True)
    wd_cols.sort(key=get_height, reverse=True)
    
    best_ws = ws_cols[0] if ws_cols else None
    best_wd = wd_cols[0] if wd_cols else None
    
    return best_ws, best_wd

def vector_average_wd(wd_series):
    """
    計算風向的向量平均 (處理 359度 與 1度 的平均問題)
    """
    rads = np.deg2rad(wd_series)
    sin_avg = np.sin(rads).mean()
    cos_avg = np.cos(rads).mean()
    avg_rad = np.arctan2(sin_avg, cos_avg)
    avg_deg = np.rad2deg(avg_rad)
    if avg_deg < 0:
        avg_deg += 360
    return avg_deg

def process_tower_files(tower_name, folder_path):
    all_data = []
    
    # 搜尋該塔所有 txt 檔案
    file_pattern = os.path.join(folder_path, f"{tower_name}*.txt")
    files = glob.glob(file_pattern)
    
    print(f"正在處理 {tower_name}，共找到 {len(files)} 個檔案...")

    for file in files:
        try:
            # 1. 預讀 Header (只讀第一行)
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                header_line = f.readline().strip().split(',')
            
            # 2. 決定要用哪兩個欄位
            target_ws, target_wd = parse_best_sensors(header_line)
            
            if not target_ws or not target_wd:
                print(f"  警告: 檔案 {os.path.basename(file)} 找不到 WS 或 WD 欄位，跳過。")
                continue
                
            # 3. 讀取資料
            # 加入 low_memory=False 避免 DtypeWarning
            cols_to_use = ['TIMESTAMP', target_ws, target_wd]
            
            df = pd.read_csv(
                file, 
                usecols=cols_to_use, 
                parse_dates=['TIMESTAMP'],
                na_values=['NaN', 'NAN', '', 'null', '-9999'], # 增加常見的缺值標記
                low_memory=False 
            )
            
            # 重新命名欄位
            df = df.rename(columns={target_ws: 'WS', target_wd: 'WD'})
            
            # ================= 修正關鍵點 =================
            # 強制將 WS 和 WD 轉為數值格式。
            # errors='coerce' 代表如果遇到無法轉換的髒資料(如文字)，直接變成 NaN
            df['WS'] = pd.to_numeric(df['WS'], errors='coerce')
            df['WD'] = pd.to_numeric(df['WD'], errors='coerce')
            # ============================================

            # 設定索引
            df.set_index('TIMESTAMP', inplace=True)
            
            # 4. 資料降頻 (Resampling)
            # 這裡要小心：如果整段都是 NaN，resample 可能會產生空的結果
            if df.empty:
                 print(f"  警告: {os.path.basename(file)} 轉換後無有效資料。")
                 continue

            df_resampled = df.resample('10min').agg({
                'WS': 'mean',
                'WD': vector_average_wd
            })
            
            all_data.append(df_resampled)
            print(f"  已讀取: {os.path.basename(file)} | 選用感測器: {target_ws}, {target_wd}")
            
        except Exception as e:
            # 印出更詳細的錯誤，方便除錯
            print(f"  讀取 {os.path.basename(file)} 失敗: {e}")

    if all_data:
        full_df = pd.concat(all_data).sort_index()
        return full_df
    else:
        return pd.DataFrame()

# ================= 執行主程式 =================

if __name__ == "__main__":
    towers = ['TGC', 'BSMI', 'TP']
    tower_datasets = {}

    for tower in towers:
        df = process_tower_files(tower, DATA_PATH)
        if not df.empty:
            tower_datasets[tower] = df
            # 儲存清洗後的資料為 CSV，方便下一步分析
            output_file = f"{tower}_Cleaned_10min.csv"
            df.to_csv(output_file)
            print(f"✅ {tower} 處理完成，已輸出至 {output_file}")
        else:
            print(f"❌ {tower} 無資料。")

    print("所有資料處理完畢。")
