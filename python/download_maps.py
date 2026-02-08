import os
from pathlib import Path
import requests
from datetime import datetime, timedelta
import time

# ================= è¨­å??€ =================

# è¨­å?ä¸‹è??„èµ·å§‹è?çµæ??¥æ?
START_DATE = datetime(2018, 4, 22)
END_DATE = datetime(2025, 3, 31)

# è¨­å?å­˜æ??¹ç›®??
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUPAN_ROOT = PROJECT_ROOT.parents[1]
#SATWND_DIR = CUPAN_ROOT / "satwnd"
SFCMAP_DIR = CUPAN_ROOT / "sfcmap"

# æ¯å¤©?„å??‹æ?æ®?
HOURS = ["00", "06", "12", "18"]

# ?©ç¨®é¡å?å°æ???URL æ¨™ç±¤?‡å?æª”å?ç¶?
# ?¼å?: (URLä¸Šç?é¡å?å­—ä¸², å­˜æ??¨ç??ç¶´, å­˜æ?è³‡æ?å¤?
IMAGE_TYPES = [
 #   ("satwnd", "satwnd", SATWND_DIR),
    ("sfcmap", "sfcmap", SFCMAP_DIR)
]

# ?½è??ç€è¦½?¨ç? Header (?¿å?è¢«æ?)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ================= ?½å?å®šç¾©?€ =================

def create_folders():
    """æª¢æŸ¥ä¸¦å»ºç«‹è??™å¤¾"""
    for _, _, folder_path in IMAGE_TYPES:
        if not Path(folder_path).exists():
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            print(f"å·²å»ºç«‹è??™å¤¾: {folder_path}")

def download_file(url, save_path):
    """ä¸‹è?æª”æ??„é€šç”¨?½å?"""
    # æª¢æŸ¥æª”æ??¯å¦å·²å???(?¿å??è?ä¸‹è?ï¼Œå¯¦?¾æ–·é»ç???
    if Path(save_path).exists():
        print(f"[?¥é?] æª”æ?å·²å??? {Path(save_path).name}")
        return

    try:
        # è¨­å? timeout ?¿å??¡ä?
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        # æª¢æŸ¥?€?‹ç¢¼ï¼?00 ä»?¡¨?å?ï¼?04 ä»?¡¨?¾ä???
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"[?å?] ä¸‹è?: {Path(save_path).name}")
        elif response.status_code == 404:
            print(f"[ç¼ºå?] ä¼ºæ??¨ç„¡æ­¤æ?æ¡?(404): {url}")
        else:
            print(f"[å¤±æ?] ?€?‹ç¢¼ {response.status_code}: {url}")
            
    except requests.exceptions.RequestException as e:
        print(f"[?¯èª¤] ????é?: {e}")

# ================= ä¸»ç?å¼åŸ·è¡Œå? =================

if __name__ == "__main__":
    create_folders()
    
    current_date = START_DATE
    
    print(f"?‹å?ä¸‹è?ä»»å?ï¼šå? {START_DATE.date()} ??{END_DATE.date()}")
    
    while current_date <= END_DATE:
        # ?†è§£?¥æ??ƒæ•¸
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")
        yyyymmdd = current_date.strftime("%Y%m%d")
        
        for hh in HOURS:
            hhmm = f"{hh}00" # è®Šæ? 0000, 0600...
            
            for url_type, prefix, folder in IMAGE_TYPES:
                # 1. çµ„å? URL
                # ç¯„ä?: https://asrad.pccu.edu.tw/catalog/cwbmap/2025/11/28/20251128_0000.cwbmap.satwnd.png
                url = f"https://asrad.pccu.edu.tw/catalog/cwbmap/{yyyy}/{mm}/{dd}/{yyyymmdd}_{hhmm}.cwbmap.{url_type}.gif"
                
                # 2. çµ„å?å­˜æ?è·¯å?
                # ?½å?ç¯„ä?: satwnd_20250128_0000.png
                filename = f"{prefix}_{yyyymmdd}_{hhmm}.png"
                save_path = Path(folder) / filename
                
                # 3. ?·è?ä¸‹è?
                download_file(url, save_path)
                
                # ç¦®è??§å»¶??(?¿å?å°ä¼º?å™¨? æ??å¤§è² æ?è¢«é? IP)
                # å¦‚æ?ä¸‹è??Ÿåº¦å¾ˆæ…¢ï¼Œå¯ä»¥æ??™å€‹è¨»è§???–è¨­å°ä?é»?
                # time.sleep(0.1) 

        # ?¥æ?? ä?å¤?
        current_date += timedelta(days=1)

    print("?€?‰ä»»?™å??ï?")


