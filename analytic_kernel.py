import yaml
import numpy as np
import cv2
import os
import sys
from time import time
from datetime import datetime
import scipy.io

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

# PyYAML이 설치되어 있어야 합니다: pip install PyYAML
# OpenCV가 설치되어 있어야 합니다: pip install opencv-python-headless


def create_height_map():
    """
    config.yaml과 닷 패턴 이미지(PSF)를 기반으로
    최종 MLA 높이 맵을 생성합니다.
    """

    # --- 1. 설정 로드 ---
    config_path = "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded config from {config_path}")
    except Exception as e:
        print(f"Error loading {config_path}: {e}")
        sys.exit(1)

    # --- 2. 파라미터 추출 및 계산 (단위: um) ---
    try:
        gen_params = config["generation_parameters"]
        exp_params = config["experiment"]

        # config.yaml에서 파라미터 읽기
        pixel_size_m = float(gen_params["pixel_size_m"])
        h_max_um = float(gen_params["height_max_um"])
        obj_dist_mm = float(gen_params["obj_dist_mm"])
        img_dist_mm = float(gen_params["img_dist_mm"])
        n = float(gen_params["n"])  # 굴절률

        # 최종 맵 크기
        img_W = int(exp_params["image_width"])
        img_H = int(exp_params["image_height"])

        # (단위 통일: um)
        px_size_um = pixel_size_m * 1e6  # (예: 2.7 um)

        # 렌즈 파라미터 계산 (이전 응답과 동일)
        # 1/f = 1/o + 1/i
        EFL_mm = 1.0 / (1.0 / obj_dist_mm + 1.0 / img_dist_mm)
        # 1/f = (n-1) * (1/R)  (plano-convex)
        ROC_mm = EFL_mm * (n - 1)

        ROC_um = ROC_mm * 1000.0  # (예: 470 um)

        # 렌즈 반경 (sag 공식)
        # r^2 = ROC^2 - (ROC - h_max)^2
        r_lens_um = np.sqrt(ROC_um**2 - (ROC_um - h_max_um) ** 2)

        print("\n--- Calculated Lenslet Parameters ---")
        print(f"Pixel Size: {px_size_um:.2f} um")
        print(f"H_max: {h_max_um:.2f} um")
        print(f"EFL: {EFL_mm:.4f} mm")
        print(f"ROC: {ROC_mm:.4f} mm ({ROC_um:.2f} um)")
        print(f"Lenslet Radius (r_lens): {r_lens_um:.2f} um")

    except KeyError as e:
        print(f"Error: Missing key {e} in {config_path}")
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

    # 구면 방정식 적용 (이전 응답과 동일)
    K = np.zeros((k_size_px, k_size_px), dtype=np.float32)
    mask = RHO_um <= r_lens_um

    # h(x,y) = (h_max - ROC) + sqrt(ROC^2 - (x^2+y^2))
    K[mask] = (h_max_um - ROC_um) + np.sqrt(ROC_um**2 - RHO_um[mask] ** 2)

    # 베이스 높이 0으로 클리핑 (수치 오류 방지)
    K[K < 0] = 0.0

    print("[OK] Lenslet kernel (K) generated.")

    # --- 4. 닷 패턴(PSF) 로드 ---
    # !!! 중요: 이 경로를 생성된 닷 패턴 .png 파일로 수정하세요 !!!
    psf_image_path = "/home/hotdog/sample/PatternGenerator/Outputs/random_dots_M_2560x1600_1344dots_mult_1.50_20251114_200646/PoissonDiskRandomDots/20251114_200646_PoissonDiskRandomDots_M_2560x1600_avg_dist_149um_mult_1.50.png"
    # (위 경로는 예시입니다. 실제 파일 경로를 넣어주세요.)

    print(f"\n--- Loading Dot Pattern (PSF) ---")
    if not os.path.exists(psf_image_path):
        print(f"Error: Dot pattern file not found at {psf_image_path}")
        print("Please update 'psf_image_path' variable in this script.")
        sys.exit(1)

    # 흑백(Grayscale)으로 이미지 로드 (0 또는 255 값)
    psf_dots = cv2.imread(psf_image_path, cv2.IMREAD_GRAYSCALE)

    if psf_dots is None:
        print(f"Error: Failed to load image from {psf_image_path}")
        sys.exit(1)

    if psf_dots.shape != (img_H, img_W):
        print(
            f"Warning: Loaded image size ({psf_dots.shape}) does not match config ({img_H}, {img_W})."
        )
        # 필요시 크기 조절 (하지만 일치해야 함)
        # psf_dots = cv2.resize(psf_dots, (img_W, img_H))

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
    # `run_generation`에서 생성한 폴더를 재사용하거나 새 폴더를 지정
    output_dir = os.path.dirname(psf_image_path)  # PSF와 같은 폴더에 저장

    # 1. 원본 데이터 저장 (float32, 단위: um)
    output_npy_path = os.path.join(
        output_dir, f"{timestamp}_{r_lens_um:.2f}um_MLA_Height_Map_meters.npy"
    )
    H_map_meters = H_map * 1e-6
    np.save(output_npy_path, H_map_meters)
    # np.save(output_npy_path, H_map)
    print(f"\nHeight map (raw float32, um) saved to: {output_npy_path}")

    output_mat_path = os.path.join(output_dir, f"{timestamp}_heightmap.mat")
    scipy.io.savemat(output_mat_path, {"map": H_map_meters})

    # 2. 시각화용 이미지 저장 (0-255 uint8)
    output_png_path = os.path.join(output_dir, "MLA_Height_Map_Visualization.png")
    H_map_vis = (H_map / h_max_um) * 255.0  # 0~h_max_um 범위를 0~255로 스케일링
    H_map_vis = H_map_vis.astype(np.uint8)
    cv2.imwrite(output_png_path, H_map_vis)
    print(f"Height map (visualization) saved to: {output_png_path}")


if __name__ == "__main__":
    create_height_map()
