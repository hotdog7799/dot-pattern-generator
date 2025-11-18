import yaml
import numpy as np
import cv2
import os
import sys
from time import time
from datetime import datetime
import scipy.io
import pandas as pd  # 엑셀 파일을 읽기 위해 pandas 추가
import glob

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

# PyYAML, OpenCV, Pandas, XlsxWriter가 필요합니다.
# pip install PyYAML opencv-python-headless pandas xlsxwriter

# -------------------------------
# [신규] 사용자 설정 모드
# -------------------------------
# MODE 1: 수동으로 경로 지정
# MODE 2: Outputs 폴더에서 가장 최신 폴더 자동 탐색
MODE = 2

# MODE 1일 경우, 1단계(run_experiments.py)에서 생성된 *하위 폴더* 경로를 지정하세요.
# (주의: 'PoissonDiskRandomDots' 폴더까지 포함해야 합니다)
# 예: "./Outputs/random_dots_M_2560x2560_1344dots_mult_1.50_20251117_220133/PoissonDiskRandomDots"
MANUAL_PATH = "./Outputs/random_dots_M_2560x2560_1344dots_mult_1.50_20251117_220133/PoissonDiskRandomDots"
# -------------------------------


def find_latest_output_files(base_dir="./Outputs"):
    """
    [MODE 2용] 지정된 Outputs 폴더에서 가장 최근에 생성된
    'random_dots_M...' 폴더를 찾아
    'pattern_settings.xlsx' 파일과 '.png' 파일 경로를 반환합니다.
    """
    try:
        list_of_dirs = glob.glob(os.path.join(base_dir, "random_dots_M_*"))
        if not list_of_dirs:
            raise FileNotFoundError(f"No 'random_dots_M_*' folders found in {base_dir}")

        latest_dir_base = max(list_of_dirs, key=os.path.getctime)
        latest_dir = os.path.join(latest_dir_base, "PoissonDiskRandomDots")

        if not os.path.exists(latest_dir):
            raise FileNotFoundError(
                f"Subfolder 'PoissonDiskRandomDots' not found in {latest_dir_base}"
            )

        print(f"Found latest output directory: {latest_dir}")

        # 1. Excel/CSV 파일 찾기
        settings_files = glob.glob(os.path.join(latest_dir, "*_settings.xlsx"))
        if not settings_files:
            excel_files = glob.glob(os.path.join(latest_dir, "*_settings.csv"))
            if not excel_files:
                raise FileNotFoundError(
                    f"No '*_settings.xlsx' or '.csv' file found in {latest_dir}"
                )
            settings_path = excel_files[0]
        else:
            settings_path = settings_files[0]

        # 2. PNG 파일 찾기
        png_files = glob.glob(os.path.join(latest_dir, "*.png"))
        if not png_files:
            raise FileNotFoundError(f"No '.png' pattern file found in {latest_dir}")

        png_path = png_files[0]

        return settings_path, png_path

    except FileNotFoundError as e:
        print(f"Error finding files: {e}")
        print("Please run 'run_experiments.py' first to generate a pattern.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while finding files: {e}")
        sys.exit(1)


def load_params_from_excel(excel_path):
    """
    pattern_settings.xlsx/csv 파일에서 파라미터를 읽어 딕셔너리로 반환합니다.
    """
    try:
        if excel_path.endswith(".xlsx"):
            df = pd.read_excel(excel_path)
        else:
            df = pd.read_csv(excel_path)

        params = df.set_index("name")["value"].to_dict()
        print(f"Successfully loaded parameters from {excel_path}")
        return params

    except Exception as e:
        print(f"Error loading parameters from {excel_path}: {e}")
        sys.exit(1)


def create_height_map():
    """
    닷 패턴(PSF)과 엑셀 설정 파일을 기반으로
    최종 MLA 높이 맵을 생성합니다.
    MODE 설정에 따라 최신 파일을 탐색하거나 수동 경로를 사용합니다.
    """

    # --- 1. [수정됨] 설정 로드 (MODE에 따라 경로 결정) ---

    psf_image_path = ""
    settings_path = ""

    if MODE == 1:
        # --- 수동 모드 ---
        print(f"--- Mode 1: Manual Path ---")
        target_dir = MANUAL_PATH
        if not os.path.isdir(target_dir):
            print(f"Error: Manual path not found: {target_dir}")
            print("Please check the 'MANUAL_PATH' variable at the top of this script.")
            sys.exit(1)

        try:
            # 1. Excel/CSV 파일 찾기
            settings_files = glob.glob(os.path.join(target_dir, "*_settings.xlsx"))
            if not settings_files:
                settings_files = glob.glob(os.path.join(target_dir, "*_settings.csv"))
            if not settings_files:
                raise FileNotFoundError(
                    f"No '*_settings.xlsx' or '.csv' file found in {target_dir}"
                )
            settings_path = settings_files[0]

            # 2. PNG 파일 찾기
            png_files = glob.glob(os.path.join(target_dir, "*.png"))
            if not png_files:
                raise FileNotFoundError(f"No '.png' pattern file found in {target_dir}")
            psf_image_path = png_files[0]

            print(f"Using manual settings: {settings_path}")
            print(f"Using manual pattern: {psf_image_path}")

        except FileNotFoundError as e:
            print(f"Error in manual path: {e}")
            sys.exit(1)

    elif MODE == 2:
        # --- 자동 최신 탐색 모드 ---
        print(f"--- Mode 2: Auto-Detect Latest ---")
        settings_path, psf_image_path = find_latest_output_files()
        print(f"Using latest settings: {settings_path}")
        print(f"Using latest pattern: {psf_image_path}")

    else:
        print(f"Error: Invalid MODE ({MODE}). Must be 1 or 2.")
        sys.exit(1)

    # --- 1.5. 파라미터 로드 ---
    params = load_params_from_excel(settings_path)

    # --- 2. 파라미터 추출 및 계산 (단위: um) ---
    try:
        # params 딕셔너리에서 파라미터 읽기 (gen_random_dot_pixel_new.py에서 저장한 값)
        pixel_size_m = float(params["Pixel_Size"])
        h_max_um = float(params["HeightProfile_Max_um"])
        obj_dist_mm = float(params["ObjectDistance_mm"])
        img_dist_mm = float(params["ImageDistance_mm"])
        n = float(params["Refractive_Index"])  # 굴절률

        # 최종 맵 크기 (엑셀에서 정사각형 크기를 읽어옴)
        img_W = int(params["M_width"])  # 예: 2560
        img_H = int(params["M_height"])  # 예: 2560

        # (단위 통일: um)
        px_size_um = pixel_size_m * 1e6  # (예: 2.7 um)

        # 렌즈 파라미터 계산
        EFL_mm = 1.0 / (1.0 / obj_dist_mm + 1.0 / img_dist_mm)
        ROC_mm = EFL_mm * (n - 1)
        ROC_um = ROC_mm * 1000.0

        # 렌즈 반경 (sag 공식)
        r_lens_um = np.sqrt(ROC_um**2 - (ROC_um - h_max_um) ** 2)

        print("\n--- Calculated Lenslet Parameters ---")
        print(f"Pixel Size: {px_size_um:.2f} um")
        print(f"H_max: {h_max_um:.2f} um")
        print(f"EFL: {EFL_mm:.4f} mm")
        print(f"ROC: {ROC_mm:.4f} mm ({ROC_um:.2f} um)")
        print(f"Lenslet Radius (r_lens): {r_lens_um:.2f} um")

    except KeyError as e:
        print(f"Error: Missing key {e} in {settings_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing parameters: {e}")
        sys.exit(1)

    # --- 3. 렌즈릿 커널(K) 생성 (단위: um) ---
    print("\n--- Generating Lenslet Kernel ---")

    # 픽셀 단위로 커널 크기 결정
    r_lens_px = int(np.ceil(r_lens_um / px_size_um))
    k_size_px = 2 * r_lens_px + 1  # 홀수 크기, 중심 픽셀 존재
    k_center_px = r_lens_px  # 중심 픽셀 인덱스

    print(f"Kernel radius: {r_lens_px} pixels")
    print(f"Kernel size: {k_size_px} x {k_size_px} pixels")

    # 커널 좌표 생성 (중심이 0, 0 / 단위: um)
    u_px = np.arange(k_size_px) - k_center_px
    u_um = u_px * px_size_um
    X_um, Y_um = np.meshgrid(u_um, u_um)
    RHO_um = np.sqrt(X_um**2 + Y_um**2)

    # 구면 방정식 적용
    K = np.zeros((k_size_px, k_size_px), dtype=np.float32)
    mask = RHO_um <= r_lens_um

    # h(x,y) = (h_max - ROC) + sqrt(ROC^2 - (x^2+y^2))
    K[mask] = (h_max_um - ROC_um) + np.sqrt(ROC_um**2 - RHO_um[mask] ** 2)

    # 베이스 높이 0으로 클리핑 (수치 오류 방지)
    K[K < 0] = 0.0

    print("[OK] Lenslet kernel (K) generated.")

    # --- 4. 닷 패턴(PSF) 로드 ---
    print(f"\n--- Loading Dot Pattern (PSF) ---")
    print(f"Loading from: {psf_image_path}")

    # 흑백(Grayscale)으로 이미지 로드 (0 또는 255 값)
    psf_dots = cv2.imread(psf_image_path, cv2.IMREAD_GRAYSCALE)

    if psf_dots is None:
        print(f"Error: Failed to load image from {psf_image_path}")
        sys.exit(1)

    # 엑셀에서 읽어온 정사각형 크기(img_H, img_W)와 비교
    if psf_dots.shape != (img_H, img_W):
        print(
            f"Error: Loaded image size ({psf_dots.shape}) does not match parameters from settings file ({img_H}, {img_W})."
        )
        sys.exit(1)

    print(f"Loaded dot pattern: {psf_dots.shape} (HxW)")

    # --- 5. 높이 맵 스탬핑 (핵심 로직) ---
    print("\n--- Generating Height Map (Stamping) ---")
    start_stamp = time()

    # 최종 높이 맵 (0으로 초기화)
    H_map = np.zeros(psf_dots.shape, dtype=np.float32)

    # 닷(dot) 위치 찾기 (값이 0보다 큰 모든 픽셀)
    dot_indices = np.argwhere(psf_dots > 0)  # (N, 2) 배열, (row, col) 순서
    num_dots = len(dot_indices)

    print(f"Found {num_dots} dots to stamp.")

    # 각 dot 위치 (iy, ix)에 대해 커널 K를 스탬핑
    for i, (iy, ix) in enumerate(dot_indices):

        if (i + 1) % 100 == 0:  # 100개마다 진행 상황 출력
            print(f"  Stamping dot {i+1}/{num_dots}...")

        # H_map에서 커널이 적용될 영역 계산 (경계 처리 포함)
        y_start = max(0, iy - k_center_px)
        y_end = min(img_H, iy + k_center_px + 1)
        x_start = max(0, ix - k_center_px)
        x_end = min(img_W, ix + k_center_px + 1)

        # H_map과 K에서 실제 슬라이싱할 인덱스 계산
        h_y_slice = slice(y_start, y_end)
        h_x_slice = slice(x_start, x_end)

        k_y_start = max(0, k_center_px - iy)
        k_y_end = k_size_px - max(0, (iy + k_center_px + 1) - img_H)
        k_x_start = max(0, k_center_px - ix)
        k_x_end = k_size_px - max(0, (ix + k_center_px + 1) - img_W)

        k_y_slice = slice(k_y_start, k_y_end)
        k_x_slice = slice(k_x_start, k_x_end)

        # *** 핵심: 겹치는 부분은 덧셈(+)이 아닌 최댓값(maximum)으로 갱신 ***
        H_map[h_y_slice, h_x_slice] = np.maximum(
            H_map[h_y_slice, h_x_slice],  # 기존 높이
            K[k_y_slice, k_x_slice],  # 새로 스탬핑할 렌즈릿 높이
        )

    end_stamp = time()
    print(f"[OK] Height map generated in {end_stamp - start_stamp:.2f} seconds.")

    # --- 6. 결과 저장 ---
    # [수정됨] `run_generation`에서 생성한 폴더를 재사용
    output_dir = os.path.dirname(psf_image_path)  # PSF와 같은 폴더에 저장

    # 엑셀 파일 이름에서 시간 부분을 따와서 파일 이름 생성
    base_filename_from_excel = os.path.basename(settings_path)
    # timestamp_from_excel = base_filename_from_excel.split("_")[0]  # 시간 부분 추출
    timestamp_from_excel = timestamp

    # 1. 원본 데이터 저장 (float32, 단위: m)
    output_npy_path = os.path.join(
        output_dir,
        f"{timestamp_from_excel}_{r_lens_um:.2f}um_MLA_Height_Map_meters.npy",
    )
    H_map_meters = H_map * 1e-6  # 미터 단위로 변환
    np.save(output_npy_path, H_map_meters)
    print(f"\nHeight map (raw float32, meters) saved to: {output_npy_path}")

    # .mat 파일로도 저장 (MATLAB 호환용)
    output_mat_path = os.path.join(output_dir, f"{timestamp_from_excel}_heightmap.mat")
    scipy.io.savemat(output_mat_path, {"map": H_map_meters})
    print(f".mat file saved to: {output_mat_path}")

    # 2. 시각화용 이미지 저장 (0-255 uint8)
    output_png_path = os.path.join(
        output_dir,
        heightmap,
        f"{timestamp_from_excel}_MLA_Height_Map_Visualization.png",
    )
    H_map_vis = (H_map / h_max_um) * 255.0  # 0~h_max_um 범위를 0~255로 스케일링
    H_map_vis = H_map_vis.astype(np.uint8)
    cv2.imwrite(output_png_path, H_map_vis)
    print(f"Height map (visualization) saved to: {output_png_path}")


if __name__ == "__main__":
    create_height_map()
