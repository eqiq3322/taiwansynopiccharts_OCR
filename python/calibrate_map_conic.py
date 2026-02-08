import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
from pathlib import Path

# ================= 設定區 =================
# 請確認圖片路徑 (建議用修復後的 PNG)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUPAN_ROOT = PROJECT_ROOT.parents[1]
IMG_PATH = CUPAN_ROOT / "sfcmap" / "sfcmap_20160301_1200.gif"

# 定義 4 個校正點 (經緯度)
# 順序：左上 -> 右上 -> 右下 -> 左下
CALIB_POINTS_LL = [
    (110.0, 40.0),  # 左上
    (140.0, 40.0),  # 右上
    (140.0, 10.0),  # 右下
    (110.0, 10.0)   # 左下
]

# LCC 投影參數 (假設東亞天氣圖標準)
# 如果網格彎曲度不對，可微調這兩個標準緯線
STD_LAT1 = 30.0
STD_LAT2 = 60.0
CENTRAL_LON = 120.0 # 中央經線 (大概抓圖的中間)

# ================= LCC 投影數學公式 =================

def lcc_forward(lat, lon):
    """將經緯度轉為 LCC 平面座標 (x, y)"""
    # 轉弧度
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat1_r = np.radians(STD_LAT1)
    lat2_r = np.radians(STD_LAT2)
    lon0_r = np.radians(CENTRAL_LON)
    
    # 計算常數
    if STD_LAT1 == STD_LAT2:
        n = np.sin(lat1_r)
    else:
        n = np.log(np.cos(lat1_r)/np.cos(lat2_r)) / \
            np.log(np.tan(np.pi/4 + lat2_r/2)/np.tan(np.pi/4 + lat1_r/2))
            
    F = (np.cos(lat1_r) * np.power(np.tan(np.pi/4 + lat1_r/2), n)) / n
    rho = F / np.power(np.tan(np.pi/4 + lat_r/2), n)
    rho0 = F / np.power(np.tan(np.pi/4 + np.radians(90)/2), n) # 北極點的 rho (如果是0則在圓心)
    
    theta = n * (lon_r - lon0_r)
    
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta) # y 軸向上為正 (北極在上方)
    
    return x, y

# ================= 主程式 =================

def fit_affine_transform(pixel_pts, lcc_pts):
    """
    計算從 LCC 平面 (x,y) 到 圖片像素 (u,v) 的仿射變換矩陣
    [u]   [a b c] [x]
    [v] = [d e f] [y]
    [1]   [0 0 1] [1]
    """
    # 使用最小平方法解 Ax = B
    # 每個點提供兩個方程式:
    # u = a*x + b*y + c
    # v = d*x + e*y + f
    
    A = []
    B = []
    for (u, v), (x, y) in zip(pixel_pts, lcc_pts):
        A.append([x, y, 1, 0, 0, 0])
        B.append(u)
        A.append([0, 0, 0, x, y, 1])
        B.append(v)
        
    A = np.array(A)
    B = np.array(B)
    
    # 解矩陣
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # 組合成 3x3 矩陣
    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0.0,       0.0,       1.0]
    ])
    return M

def draw_curved_grid(img, M_lcc_to_pix):
    h, w = img.shape[:2]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.title("LCC Conic Grid Verification (Curved Lines)")

    # 畫經線 (Longitude) - 輻射狀直線
    for lon in range(100, 155, 5):
        lats = np.linspace(0, 60, 100) # 切多一點點讓線條滑順
        xs, ys = [], []
        for lat in lats:
            lx, ly = lcc_forward(lat, lon)
            # 轉換到 Pixel
            vec = np.dot(M_lcc_to_pix, [lx, ly, 1])
            xs.append(vec[0])
            ys.append(vec[1])
        plt.plot(xs, ys, 'b-', alpha=0.5, linewidth=0.8)
        # 標示
        if 0 <= xs[50] < w and 0 <= ys[50] < h:
            plt.text(xs[50], ys[50], f"{lon}E", color='blue', fontsize=8, weight='bold')

    # 畫緯線 (Latitude) - 彎曲弧線
    for lat in range(0, 65, 5):
        lons = np.linspace(90, 160, 100)
        xs, ys = [], []
        for lon in lons:
            lx, ly = lcc_forward(lat, lon)
            vec = np.dot(M_lcc_to_pix, [lx, ly, 1])
            xs.append(vec[0])
            ys.append(vec[1])
        plt.plot(xs, ys, 'r-', alpha=0.5, linewidth=0.8)
        # 標示
        if 0 <= xs[50] < w and 0 <= ys[50] < h:
            plt.text(xs[50], ys[50], f"{lat}N", color='red', fontsize=8, weight='bold')

    plt.show()

if __name__ == "__main__":
    if not IMG_PATH.exists():
        print("圖片不存在")
        exit()
        
    img = mpimg.imread(str(IMG_PATH))
    
    print("=== LCC 圓錐投影校正 ===")
    print("請依照順序點擊這 4 個點:")
    for i, p in enumerate(CALIB_POINTS_LL):
        print(f"{i+1}. {p} (經度, 緯度)")
        
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title("Click 4 Points (1.TopLeft -> 2.TopRight -> 3.BotRight -> 4.BotLeft)")
    pts = plt.ginput(n=4, timeout=-1)
    plt.close()

    if len(pts) == 4:
        # 1. 將 4 個目標經緯度轉成 LCC 座標
        lcc_pts = []
        for (lon, lat) in CALIB_POINTS_LL:
            lx, ly = lcc_forward(lat, lon)
            lcc_pts.append((lx, ly))
            
        # 2. 計算轉換矩陣 (LCC -> Pixel)
        M = fit_affine_transform(pts, lcc_pts)
        
        # 3. 畫圖驗證
        print("繪製網格中...")
        draw_curved_grid(img, M)
        
        # 4. 輸出給主程式用的參數
        print("\n" + "="*40)
        print("【校正成功】請將以下整段程式碼複製到您的主程式最上方：")
        print("-" * 40)
        print("# === LCC 投影參數 (由校正程式生成) ===")
        print(f"LCC_M = np.array({np.array2string(M, separator=', ')})")
        print(f"LCC_M_INV = np.linalg.inv(LCC_M)")
        print(f"STD_LAT1 = {STD_LAT1}")
        print(f"STD_LAT2 = {STD_LAT2}")
        print(f"CENTRAL_LON = {CENTRAL_LON}")
        print("-" * 40)
