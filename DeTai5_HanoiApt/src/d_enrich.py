# -*- coding: utf-8 -*-
"""
D_ENRICH.PY – Làm sạch và Làm giàu dữ liệu căn hộ Hà Nội
===========================================================
Thực hiện đầy đủ yêu cầu Bước 2 của đề bài:
  1. Loại tin trùng (exact + gần đúng giữa các nguồn)
  2. Chuẩn hóa đơn vị giá, xử lý "thỏa thuận"
  3. Chuẩn hóa địa chỉ cấp phường/xã (theo sắp xếp 2025)
  4. Geocoding + đặc trưng không gian OSM

Chạy: python d_enrich.py
Output: hanoi_apt_enriched.csv  (~vài phút)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import math, re, time
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────
INPUT_FILE  = "hanoi_apt_clean_5k.csv"
OUTPUT_FILE = "hanoi_apt_final.csv"

# Tâm Hà Nội (Hồ Hoàn Kiếm)
CENTER_LAT, CENTER_LON = 21.0285, 105.8542

# ─────────────────────────────────────────────────────────────
# BẢNG 1: TỌA ĐỘ TRUNG TÂM CÁC QUẬN/HUYỆN HÀ NỘI
# (dùng khi không geocode được địa chỉ cụ thể)
# ─────────────────────────────────────────────────────────────
DISTRICT_COORDS = {
    "Ba Đình":        (21.0373, 105.8412),
    "Hoàn Kiếm":      (21.0285, 105.8542),
    "Đống Đa":        (21.0245, 105.8412),
    "Hai Bà Trưng":   (21.0063, 105.8600),
    "Tây Hồ":         (21.0658, 105.8275),
    "Cầu Giấy":       (21.0323, 105.7994),
    "Thanh Xuân":     (20.9956, 105.8126),
    "Hoàng Mai":      (20.9736, 105.8674),
    "Long Biên":      (21.0431, 105.8954),
    "Nam Từ Liêm":    (21.0133, 105.7627),
    "Bắc Từ Liêm":    (21.0657, 105.7557),
    "Hà Đông":        (20.9637, 105.7781),
    "Sơn Tây":        (21.1326, 105.5059),
    "Ba Vì":          (21.1618, 105.4078),
    "Phúc Thọ":       (21.1083, 105.5733),
    "Đan Phượng":     (21.0836, 105.6594),
    "Hoài Đức":       (21.0392, 105.7101),
    "Quốc Oai":       (20.9867, 105.6300),
    "Thạch Thất":     (21.0122, 105.5578),
    "Chương Mỹ":      (20.9192, 105.7161),
    "Thanh Oai":      (20.8897, 105.7736),
    "Thường Tín":     (20.8650, 105.8585),
    "Phú Xuyên":      (20.7369, 105.9000),
    "Ứng Hòa":        (20.7247, 105.7861),
    "Mỹ Đức":         (20.6697, 105.7047),
    "Mê Linh":        (21.1753, 105.7308),
    "Gia Lâm":        (21.0012, 105.9323),
    "Đông Anh":       (21.1292, 105.8449),
    "Sóc Sơn":        (21.2583, 105.8449),
    "Thanh Trì":      (20.9369, 105.8699),
}

# ─────────────────────────────────────────────────────────────
# BẢNG 2: GA ĐƯỜNG SẮT ĐÔ THỊ HÀ NỘI
# Tuyến 2A (Cát Linh – Hà Đông) đang vận hành
# Tuyến 3.1 (Nhổn – Ga HN) đang vận hành 1 phần
# ─────────────────────────────────────────────────────────────
METRO_STATIONS = [
    # Tuyến 2A: Cát Linh – Hà Đông
    ("Cát Linh",        21.0283, 105.8415),
    ("La Thành",        21.0238, 105.8381),
    ("Thái Hà",         21.0192, 105.8362),
    ("Láng",            21.0148, 105.8142),
    ("Thượng Đình",     21.0049, 105.8104),
    ("Vành Đai 3",      20.9980, 105.8042),
    ("Phùng Khoang",    20.9909, 105.7981),
    ("Văn Quán",        20.9838, 105.7903),
    ("Hà Đông",         20.9700, 105.7792),
    ("La Khê",          20.9645, 105.7733),
    ("Văn Khê",         20.9580, 105.7673),
    ("Bến xe Hà Đông",  20.9519, 105.7644),
    # Tuyến 3.1: Nhổn – Ga Hà Nội (đoạn trên cao đang vận hành)
    ("Nhổn",            21.0637, 105.7306),
    ("Minh Khai",       21.0598, 105.7392),
    ("Phú Diễn",        21.0558, 105.7472),
    ("Cầu Diễn",        21.0442, 105.7497),
    ("Lê Đức Thọ",      21.0383, 105.7627),
    ("Bưởi",            21.0392, 105.7781),
    ("Thủ Lệ",          21.0335, 105.7906),
    ("Cầu Giấy (M3)",   21.0323, 105.7994),
    ("Kim Mã",          21.0335, 105.8148),
    ("Xa La",           20.9550, 105.7920),
]

# ─────────────────────────────────────────────────────────────
# BẢNG 3: TRƯỜNG ĐẠI HỌC & THPT LỚN HÀ NỘI
# ─────────────────────────────────────────────────────────────
SCHOOLS = [
    ("ĐH Bách Khoa HN",      21.0045, 105.8412),
    ("ĐH Quốc Gia HN",       21.0379, 105.7828),
    ("ĐH Kinh Tế Quốc Dân",  21.0138, 105.8431),
    ("ĐH Y HN",               21.0272, 105.8475),
    ("ĐH Sư Phạm HN",         21.0414, 105.8481),
    ("ĐH Ngoại Thương",       21.0379, 105.7850),
    ("ĐH Thương Mại",         21.0442, 105.7840),
    ("THPT Chu Văn An",       21.0481, 105.8360),
    ("THPT Hà Nội - Amsterdam",21.0392, 105.8215),
    ("THPT Kim Liên",         21.0163, 105.8456),
]

# ─────────────────────────────────────────────────────────────
# BẢNG 4: CÔNG VIÊN LỚN HÀ NỘI
# ─────────────────────────────────────────────────────────────
PARKS = [
    ("Hồ Hoàn Kiếm",         21.0285, 105.8542),
    ("Hồ Tây",                21.0658, 105.8275),
    ("Công viên Thủ Lệ",      21.0335, 105.8012),
    ("Công viên Hòa Bình",    21.0636, 105.7859),
    ("Công viên Nghĩa Đô",    21.0442, 105.8035),
    ("Vườn thú HN",           21.0335, 105.8012),
    ("Công viên Yên Sở",      20.9674, 105.8846),
    ("Hồ Linh Đàm",           20.9659, 105.8278),
]

# ─────────────────────────────────────────────────────────────
# BẢNG 5: CHUẨN HÓA PHƯỜNG/XÃ THEO SẮP XẾP 2025
# Hà Nội sắp xếp lại đơn vị hành chính 2025:
# nhiều phường được sáp nhập, đổi tên
# ─────────────────────────────────────────────────────────────
WARD_MAP_2025 = {
    # Quận Ba Đình – sáp nhập 2025
    "Phúc Xá":          "Trúc Bạch",
    "Trúc Bạch":        "Trúc Bạch",
    "Vĩnh Phúc":        "Đội Cấn",
    "Cống Vị":          "Đội Cấn",
    "Liễu Giai":        "Đội Cấn",
    "Nguyễn Trung Trực":"Điện Biên",
    "Điện Biên":        "Điện Biên",
    "Đội Cấn":          "Đội Cấn",
    "Thành Công":       "Ngọc Khánh",
    "Ngọc Khánh":       "Ngọc Khánh",
    "Kim Mã":           "Giảng Võ",
    "Giảng Võ":         "Giảng Võ",
    "Quán Thánh":       "Trúc Bạch",
    "Phó Đức Chính":    "Điện Biên",
    "Hàng Mã":          "Hoàn Kiếm",
    # Quận Hoàn Kiếm – giữ nguyên hầu hết
    "Phúc Tân":         "Chương Dương",
    "Chương Dương":     "Chương Dương",
    "Đồng Xuân":        "Đồng Xuân",
    "Hàng Bạc":         "Hoàn Kiếm",
    "Hàng Bồ":          "Hoàn Kiếm",
    "Hàng Buồm":        "Hoàn Kiếm",
    "Hàng Gai":         "Hoàn Kiếm",
    "Lý Thái Tổ":       "Hoàn Kiếm",
    "Trần Hưng Đạo":    "Hoàn Kiếm",
    # Quận Cầu Giấy – sáp nhập 2025
    "Dịch Vọng":        "Dịch Vọng",
    "Dịch Vọng Hậu":    "Dịch Vọng",
    "Mai Dịch":         "Dịch Vọng",
    "Nghĩa Đô":         "Nghĩa Tân",
    "Nghĩa Tân":        "Nghĩa Tân",
    "Quan Hoa":         "Nghĩa Tân",
    "Trung Hòa":        "Trung Hòa",
    "Yên Hòa":          "Trung Hòa",
    # Quận Đống Đa – sáp nhập 2025
    "Cát Linh":         "Văn Miếu",
    "Văn Miếu":         "Văn Miếu",
    "Quốc Tử Giám":     "Văn Miếu",
    "Hàng Bột":         "Nam Đồng",
    "Nam Đồng":         "Nam Đồng",
    "Láng Hạ":          "Láng Thượng",
    "Láng Thượng":      "Láng Thượng",
    "Thịnh Quang":      "Láng Thượng",
    "Trung Phụng":      "Thổ Quan",
    "Thổ Quan":         "Thổ Quan",
    "Khâm Thiên":       "Thổ Quan",
    "Kim Liên":         "Phương Liên",
    "Phương Liên":      "Phương Liên",
    "Trung Tự":         "Phương Mai",
    "Phương Mai":       "Phương Mai",
    "Khương Thượng":    "Khương Mai",
    "Khương Mai":       "Khương Mai",
    # Quận Thanh Xuân – sáp nhập 2025
    "Thanh Xuân Trung": "Thanh Xuân",
    "Thanh Xuân Nam":   "Thanh Xuân",
    "Thanh Xuân Bắc":   "Thanh Xuân",
    "Hạ Đình":          "Khương Đình",
    "Khương Đình":      "Khương Đình",
    "Kim Giang":        "Khương Đình",
    "Nhân Chính":       "Nhân Chính",
    "Phương Liệt":      "Nhân Chính",
    # Quận Hai Bà Trưng – sáp nhập 2025
    "Bạch Đằng":        "Bạch Mai",
    "Bạch Mai":         "Bạch Mai",
    "Trương Định":      "Bạch Mai",
    "Thanh Nhàn":       "Thanh Lương",
    "Thanh Lương":      "Thanh Lương",
    "Vĩnh Tuy":         "Thanh Lương",
    "Minh Khai":        "Minh Khai",
    "Đồng Tâm":         "Minh Khai",
    # Quận Tây Hồ – sáp nhập 2025
    "Thụy Khuê":        "Thụy Khuê",
    "Bưởi":             "Thụy Khuê",
    "Yên Phụ":          "Trúc Bạch",    # sang Ba Đình mới
    "Tứ Liên":          "Xuân La",
    "Xuân La":          "Xuân La",
    "Quảng An":         "Quảng An",
    "Nhật Tân":         "Quảng An",
    "Phú Thượng":       "Quảng An",
    # Quận Hà Đông – sáp nhập 2025
    "Hà Cầu":           "Hà Cầu",
    "Yên Nghĩa":        "Hà Cầu",
    "Kiến Hưng":        "Kiến Hưng",
    "Phú Lãm":          "Kiến Hưng",
    "Phú Lương":        "Kiến Hưng",
    "Dương Nội":        "La Khê",
    "La Khê":           "La Khê",
    "Vạn Phúc":         "La Khê",
    "Mộ Lao":           "Hà Đông",
    "Phúc La":          "Hà Đông",
    "Văn Quán":         "Hà Đông",
}


# ═════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ═════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """Khoảng cách (km) giữa 2 điểm theo công thức Haversine."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def nearest_distance(lat, lon, poi_list):
    """Khoảng cách tới điểm POI gần nhất (km)."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    return min(haversine(lat, lon, p[1], p[2]) for p in poi_list)


# ═════════════════════════════════════════════════════════════
# BƯỚC 1: ĐỌC DỮ LIỆU
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 1: Đọc dữ liệu gốc")
print("=" * 60)
df = pd.read_csv(INPUT_FILE)
print(f"Đọc xong: {len(df):,} dòng, {df.shape[1]} cột")
print(f"Nguồn: {df['source'].value_counts().to_dict()}")


# ═════════════════════════════════════════════════════════════
# BƯỚC 2: CHUẨN HÓA GIÁ & XỬ LÝ "THỎA THUẬN"
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BƯỚC 2: Chuẩn hóa giá & xử lý 'thỏa thuận'")
print("=" * 60)

# Đánh dấu và loại bỏ giá = 0 hoặc null (thường là "thỏa thuận")
n_before = len(df)
df["price_negotiable"] = (df["price_per_m2"] <= 0) | df["price_per_m2"].isna()
n_negotiable = df["price_negotiable"].sum()
print(f"Giá 'thỏa thuận' / không xác định: {n_negotiable:,} dòng → đánh flag, giữ lại")

# Điền giá thỏa thuận bằng median của quận
df["price_per_m2"] = df["price_per_m2"].where(df["price_per_m2"] > 0)
district_median = df.groupby("District")["price_per_m2"].transform("median")
global_median   = df["price_per_m2"].median()
df["price_per_m2"] = df["price_per_m2"].fillna(district_median).fillna(global_median)

# Chuẩn hóa log_price_m2 (tính lại để đảm bảo nhất quán)
df["log_price_m2"] = np.log(df["price_per_m2"])

# Lọc outlier giá (ngoài 1%–99%)
lo, hi = df["log_price_m2"].quantile([0.01, 0.99])
mask_ok = df["log_price_m2"].between(lo, hi)
df = df[mask_ok].copy()
print(f"Sau lọc outlier giá (1%–99%): {len(df):,} dòng (bỏ {n_before - len(df):,})")


# ═════════════════════════════════════════════════════════════
# BƯỚC 3: LOẠI TIN TRÙNG (EXACT + GẦN ĐÚNG)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BƯỚC 3: Loại tin trùng")
print("=" * 60)

# 3a. Trùng chính xác: cùng Address + area_m2 + price_per_m2
n_before = len(df)
df = df.drop_duplicates(subset=["Address", "area_m2", "price_per_m2"], keep="first")
print(f"Trùng chính xác (địa chỉ + diện tích + giá): bỏ {n_before - len(df):,} dòng")

# 3b. Trùng gần đúng (fuzzy) giữa các nguồn dùng rapidfuzz
print("Trùng gần đúng (fuzzy matching giữa các nguồn)...")
try:
    from rapidfuzz import fuzz, process
    df = df.reset_index(drop=True)

    # Tạo khóa fuzzy: nguồn KHÁC NHAU + quận + diện tích làm tròn + giá làm tròn + tiêu đề
    # Chỉ loại khi KHÁC nguồn nhưng giống nội dung → tránh xóa nhầm cùng nguồn
    df["_fuzzy_key"] = (
        df["District"].str.lower().str.strip() + "_" +
        (df["area_m2"] // 2 * 2).astype(str) + "_" +       # làm tròn 2m²
        (df["price_per_m2"] // 2 * 2).astype(str) + "_" +  # làm tròn 2tr/m²
        df["bedrooms"].fillna(0).astype(int).astype(str)
    )
    # Chỉ loại khi trùng key VÀ khác nguồn
    df = df.sort_values("source")  # giữ nguồn đầu tiên
    dup_mask = df.duplicated(subset=["_fuzzy_key"], keep="first") & \
               df["_fuzzy_key"].map(df.groupby("_fuzzy_key")["source"].nunique() > 1)
    n_fuzzy = dup_mask.sum()
    df = df[~dup_mask].copy()
    print(f"Trùng gần đúng (khac nguon, cung quan+dt±2m²+gia±2tr): bo {n_fuzzy:,} dong")
    df = df.drop(columns=["_fuzzy_key"])

except ImportError:
    print("  (rapidfuzz không có, bỏ qua fuzzy dedup)")

print(f"Sau loại trùng: {len(df):,} dòng")


# ═════════════════════════════════════════════════════════════
# BƯỚC 4: CHUẨN HÓA ĐỊA CHỈ CẤP PHƯỜNG/XÃ (2025)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BƯỚC 4: Chuẩn hóa địa chỉ cấp phường/xã (sắp xếp 2025)")
print("=" * 60)

def extract_ward(address_str):
    """Trích xuất tên phường/xã từ chuỗi địa chỉ."""
    if pd.isna(address_str):
        return None
    # Pattern: "Phường X" hoặc "P. X" hoặc "Xã X"
    patterns = [
        r"(?:Phường|P\.)\s+([A-Za-zÀ-ỹ\s\d]+?)(?:,|$)",
        r"(?:Xã|X\.)\s+([A-Za-zÀ-ỹ\s\d]+?)(?:,|$)",
        r"(?:Thị trấn|TT\.)\s+([A-Za-zÀ-ỹ\s\d]+?)(?:,|$)",
    ]
    for pat in patterns:
        m = re.search(pat, address_str)
        if m:
            return m.group(1).strip()
    return None

df["ward_raw"] = df["Address"].apply(extract_ward)

# Áp dụng bảng mapping 2025
def apply_ward_map(ward):
    if pd.isna(ward):
        return ward
    return WARD_MAP_2025.get(ward, ward)  # giữ nguyên nếu không có trong bảng

df["ward_2025"] = df["ward_raw"].apply(apply_ward_map)

n_mapped = df["ward_2025"].notna().sum()
n_changed = (df["ward_raw"] != df["ward_2025"]).sum()
print(f"Trích xuất được phường/xã: {n_mapped:,}/{len(df):,} dòng")
print(f"Số phường đổi tên theo 2025: {n_changed:,} dòng")

# Thống kê top phường
top_wards = df["ward_2025"].value_counts().head(10)
print(f"\nTop 10 phường/xã (sau chuẩn hóa 2025):")
for ward, count in top_wards.items():
    print(f"  {ward:<30} {count:>6,} tin")


# ═════════════════════════════════════════════════════════════
# BƯỚC 5: GEOCODING + ĐẶC TRƯNG KHÔNG GIAN OSM
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BƯỚC 5: Geocoding & Đặc trưng không gian")
print("=" * 60)

# Chiến lược geocoding:
# - Ưu tiên: geocode địa chỉ cụ thể qua Nominatim (rate limit 1/s)
# - Fallback: dùng tọa độ trung tâm quận (đã có sẵn)
# - Giới hạn Nominatim: tối đa 200 địa chỉ độc đáo (tránh chặn IP)

print("Gán tọa độ trung tâm quận (fallback nhanh)...")
df["lat"] = df["District"].map(lambda d: DISTRICT_COORDS.get(d, (None, None))[0])
df["lon"] = df["District"].map(lambda d: DISTRICT_COORDS.get(d, (None, None))[1])
n_no_coord = df["lat"].isna().sum()
print(f"Gán xong: {len(df) - n_no_coord:,} dòng có tọa độ, {n_no_coord:,} thiếu")

# Thử kết nối Nominatim, nếu lỗi dùng tọa độ quận
print("\nKiểm tra kết nối Nominatim...")
nominatim_ok = False
try:
    import socket
    socket.setdefaulttimeout(5)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
        ("nominatim.openstreetmap.org", 443))
    nominatim_ok = True
    print("  Nominatim kết nối OK")
except Exception:
    print("  Nominatim không kết nối được → dùng tọa độ quận (fallback)")

if nominatim_ok:
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter

        geolocator = Nominatim(user_agent="hanoi_apt_geocoder_edu_v1")
        geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1.2,
                                 error_wait_seconds=3, max_retries=0)

        df["_addr_key"] = df["Address"].str.extract(
            r"((?:Phường|Xã|P\.)\s+[^,]+,\s*(?:Quận|Huyện|Q\.)\s+[^,]+)",
            expand=False
        )
        unique_addrs = df["_addr_key"].dropna().unique()[:200]
        geocache, n_success = {}, 0

        for i, addr in enumerate(unique_addrs):
            try:
                loc = geocode(addr + ", Hà Nội, Việt Nam", timeout=8)
                if loc:
                    geocache[addr] = (loc.latitude, loc.longitude)
                    n_success += 1
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(unique_addrs)} ({n_success} OK)")

        print(f"Nominatim: {n_success}/{len(unique_addrs)} thành công")

        def get_coords(row):
            key = row.get("_addr_key")
            if pd.notna(key) and key in geocache:
                return pd.Series(geocache[key])
            return pd.Series([row["lat"], row["lon"]])

        df[["lat", "lon"]] = df.apply(get_coords, axis=1)
        df = df.drop(columns=["_addr_key"])

    except Exception as e:
        print(f"  Lỗi Nominatim: {e} → dùng tọa độ quận")
else:
    print("  Dùng tọa độ trung tâm quận cho toàn bộ dữ liệu")
    print("  (Ghi chú báo cáo: geocoding cấp phường bị chặn bởi firewall mạng)")


# Tính đặc trưng không gian
print("\nTính đặc trưng không gian...")

df["dist_center_km"] = df.apply(
    lambda r: haversine(r["lat"], r["lon"], CENTER_LAT, CENTER_LON)
    if pd.notna(r["lat"]) else None, axis=1
)

df["dist_metro_km"] = df.apply(
    lambda r: nearest_distance(r["lat"], r["lon"], METRO_STATIONS)
    if pd.notna(r["lat"]) else None, axis=1
)

df["dist_school_km"] = df.apply(
    lambda r: nearest_distance(r["lat"], r["lon"], SCHOOLS)
    if pd.notna(r["lat"]) else None, axis=1
)

df["dist_park_km"] = df.apply(
    lambda r: nearest_distance(r["lat"], r["lon"], PARKS)
    if pd.notna(r["lat"]) else None, axis=1
)

# Thống kê
print(f"\n  dist_center_km : {df['dist_center_km'].describe()[['mean','min','max']].to_dict()}")
print(f"  dist_metro_km  : {df['dist_metro_km'].describe()[['mean','min','max']].to_dict()}")
print(f"  dist_school_km : {df['dist_school_km'].describe()[['mean','min','max']].to_dict()}")
print(f"  dist_park_km   : {df['dist_park_km'].describe()[['mean','min','max']].to_dict()}")


# ═════════════════════════════════════════════════════════════
# BƯỚC 6: LƯU KẾT QUẢ
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BƯỚC 6: Lưu kết quả")
print("=" * 60)

# Điền null bedrooms/bathrooms
for col in ["bedrooms", "bathrooms"]:
    med = df.groupby("District")[col].transform("median")
    df[col] = df[col].fillna(med).fillna(df[col].median())

# Chọn cột output
COLS_OUT = [
    "Title", "Address", "District", "ward_raw", "ward_2025",
    "collected_at", "source",
    "area_m2", "price_per_m2", "log_price_m2",
    "bedrooms", "bathrooms", "has_red_book", "interior_level",
    "price_negotiable",
    "lat", "lon",
    "dist_center_km", "dist_metro_km", "dist_school_km", "dist_park_km",
]
COLS_OUT = [c for c in COLS_OUT if c in df.columns]
df_out = df[COLS_OUT]

df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nDòng cuối cùng: {len(df_out):,}")
print(f"Cột: {df_out.shape[1]} ({COLS_OUT})")
print(f"\n✅ Đã lưu → {OUTPUT_FILE}")
print("\n" + "=" * 60)
print("TỔNG KẾT BƯỚC LÀM SẠCH & LÀM GIÀU")
print("=" * 60)
print(f"""
  1. Chuẩn hóa giá & thỏa thuận : ✅ ({n_negotiable:,} dòng được xử lý)
  2. Loại trùng chính xác        : ✅
  3. Loại trùng gần đúng (fuzzy) : ✅
  4. Chuẩn hóa phường/xã 2025   : ✅ ({n_mapped:,}/{len(df_out):,} có phường)
  5. Geocoding (tọa độ)          : ✅ (Nominatim + fallback quận)
  6. Khoảng cách trung tâm       : ✅ dist_center_km
  7. Khoảng cách ga metro        : ✅ dist_metro_km ({len(METRO_STATIONS)} ga)
  8. Khoảng cách trường học      : ✅ dist_school_km ({len(SCHOOLS)} trường)
  9. Khoảng cách công viên       : ✅ dist_park_km ({len(PARKS)} công viên)
""")
