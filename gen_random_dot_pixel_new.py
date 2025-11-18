import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import cv2
import matplotlib.pyplot as plt
from time import time
from scipy.spatial import Voronoi
from idaes.core.surrogate.pysmo.sampling import CVTSampling
import datetime
import math
import numpy as np
import math
import os, csv, datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}\n".format(DEVICE))


class RandomDotsGenerator:
    def __init__(
        self,
        target_size=(750, 750),
        avg_distance_um=360,
        pixel_size=2.7e-6,
        visualize=False,
    ):
        # target_size: tuple (width, height) 또는 int -> square
        if isinstance(target_size, int):
            self.width = int(target_size)
            self.height = int(target_size)
        else:
            self.width = int(target_size[0])
            self.height = int(target_size[1])

        self.targetSize = (self.width, self.height)  # keep for backwards compat
        self.avg_distance_um = avg_distance_um
        self.pixel_size = pixel_size
        self.visualize = visualize
        self.start_time = time()

        self.numDots = self.calculate_num_dots()

    def calculate_num_dots(self):
        area_meters = (self.width * self.pixel_size) * (self.height * self.pixel_size)
        avg_distance_meters = self.avg_distance_um * 1e-6
        numDots = int(area_meters / (avg_distance_meters**2))
        print("Estimated number of dots:", numDots)
        return numDots

    def makeDistantRandomDots(self):
        print("Generating Random Dots with GPU... \n")
        numDots = self.numDots
        targetSize = self.targetSize

        # Calculate the number of dots in one dimension based on the square root of numDots
        numDots1D = int(torch.sqrt(torch.tensor(numDots, dtype=torch.float32)).item())

        # Create boundaries for dot placement, spread evenly across the target size
        boundaries = torch.linspace(0, targetSize, numDots1D + 1, device=DEVICE)

        # Generate random dot positions on GPU using PyTorch (between the size of one boundary)
        dots = torch.randint(0, int(boundaries[1]), (numDots * 2,), device=DEVICE)

        # Initialize target dot positions
        targetDots = torch.zeros((numDots, 2), dtype=torch.int32, device=DEVICE)
        for i in range(numDots):
            targetDots[i, 0] = dots[2 * i]
            targetDots[i, 1] = dots[2 * i + 1]

        # Adjust the dot positions to span across the entire grid
        DotsInCoordinate = targetDots.clone()
        for k in range(2):
            for i in range(numDots1D):
                for j in range(numDots1D):
                    index = i * numDots1D + j
                    if index < numDots:  # Prevent indexing overflow
                        DotsInCoordinate[index, k] = (
                            targetDots[index, k] + boundaries[j * (1 - k) + i * k]
                        )

        # Create the PSF (target plane) on GPU and assign dot positions with a value of 255
        targetPSF = torch.zeros(
            (targetSize, targetSize), device=DEVICE, dtype=torch.uint8
        )
        targetPSF[DotsInCoordinate[:, 0], DotsInCoordinate[:, 1]] = (
            255  # Set dot value to 255
        )

        # Convert the PSF to numpy for further processing or saving
        targetPSF = targetPSF.cpu().numpy()

        end_time = time()
        print("Done: {:.2f}s\n".format(end_time - self.start_time))

        return targetPSF

    def VoronoiRandomDots(self):
        print("Generating Voronoi Random Dots with GPU... \n")
        # Generate random points on GPU
        randArr = (
            torch.randint(0, self.targetSize, (self.numDots, 2), device=DEVICE)
            .cpu()
            .numpy()
        )
        vor = Voronoi(randArr)

        # Create the PSF using points
        initPSF = torch.zeros(
            (self.targetSize, self.targetSize), device=DEVICE, dtype=torch.uint8
        )
        for i in range(self.numDots):
            coordinates = torch.round(torch.tensor(vor.points[i])).long().to(DEVICE)
            initPSF[coordinates[0], coordinates[1]] = 255  # Set dot value to 255

        targetPSF = torch.flip(initPSF, dims=[1]).T
        targetPSF = targetPSF.cpu().numpy()

        end_time = time()
        print("Done: {:.2f}s\n".format(end_time - self.start_time))

        return targetPSF

    def CentroidVoronoiRandomDots(self, tolerance=1e-7):
        print("Generating Centroid Voronoi Random Dots with GPU... \n")
        numDots = self.numDots
        targetSize = self.targetSize

        inputList = [[0, 0], [targetSize, targetSize]]
        b = CVTSampling(
            inputList, numDots, tolerance=tolerance, sampling_type="creation"
        )
        samples = torch.tensor(b.sample_points(), dtype=torch.float32, device=DEVICE)

        # Create the Voronoi diagram using sampled points
        samples_np = samples.cpu().numpy()  # Voronoi does not support PyTorch tensors
        vor = Voronoi(samples_np)

        # Create the PSF using points
        initPSF = torch.zeros(
            (targetSize, targetSize), device=DEVICE, dtype=torch.uint8
        )
        for i in range(numDots):
            coordinates = torch.round(torch.tensor(vor.points[i])).long().to(DEVICE)
            initPSF[coordinates[0], coordinates[1]] = 255  # Set dot value to 255

        targetPSF = torch.flip(initPSF, dims=[1]).T
        targetPSF = targetPSF.cpu().numpy()

        end_time = time()
        print("Done: {:.2f}s\n".format(end_time - self.start_time))

        return targetPSF

    # PoissonDiskRandomDots 내부에서 targetSize 관련 부분 전부 교체
    def PoissonDiskRandomDots(
        self, min_distance=None, max_attempts=100, pixel_strict=True, neighbor_margin=2
    ):
        print("Generating Poisson Disk Random Dots...\n")

        if min_distance is None:
            min_distance = self.avg_distance_um / (self.pixel_size * 1e6)

        r = float(min_distance)
        # 연산용 width, height
        W = int(self.width)
        H = int(self.height)

        # cell_size 기준은 픽셀 단위 거리 r 에 대해 설정 (r: 픽셀)
        cell_size = r / math.sqrt(2.0)
        gw = int(math.ceil(W / cell_size))
        gh = int(math.ceil(H / cell_size))
        grid = torch.full((gw, gh), -1, dtype=torch.int32, device=DEVICE)

        samples = []
        samples_px = []
        active = []

        def in_bounds_xy(p):
            return (0 <= p[0] < W) and (0 <= p[1] < H)

        def xy_to_grid(p):
            return int((p[0] / cell_size).item()), int((p[1] / cell_size).item())

        def rc_to_grid(rc):
            gx = int((rc[1] / cell_size))
            gy = int((rc[0] / cell_size))
            return gx, gy

        # 첫 점: 중앙
        p0 = torch.tensor([W // 2, H // 2], device=DEVICE, dtype=torch.float32)
        samples.append(p0)
        if pixel_strict:
            row = int(torch.clamp(p0[1].round(), 0, H - 1).item())
            col = int(torch.clamp(p0[0].round(), 0, W - 1).item())
            samples_px.append((row, col))
            gx, gy = rc_to_grid((row, col))
        else:
            gx, gy = xy_to_grid(p0)

        grid[gx, gy] = 0
        active.append(0)

        while active:
            aidx = int(torch.randint(0, len(active), (1,), device=DEVICE).item())
            sidx = active[aidx]
            center = samples[sidx]

            found = False
            for _ in range(max_attempts):
                u = torch.rand(1, device=DEVICE)
                rad = torch.sqrt(((np.sqrt(3) * r) ** 2 - (r * r)) * u + r * r)
                angle = 2.0 * math.pi * torch.rand(1, device=DEVICE)
                dir2 = torch.stack((torch.cos(angle), torch.sin(angle))).squeeze(-1)
                p = center + dir2 * rad

                if not in_bounds_xy(p):
                    continue

                if pixel_strict:
                    row = int(torch.clamp(p[1].round(), 0, H - 1).item())
                    col = int(torch.clamp(p[0].round(), 0, W - 1).item())
                    gx = int((col / cell_size))
                    gy = int((row / cell_size))
                    ix0 = max(gx - neighbor_margin, 0)
                    ix1 = min(gx + neighbor_margin, gw - 1)
                    iy0 = max(gy - neighbor_margin, 0)
                    iy1 = min(gy + neighbor_margin, gh - 1)

                    ok = True
                    for ix in range(ix0, ix1 + 1):
                        for iy in range(iy0, iy1 + 1):
                            gval = grid[ix, iy].item()
                            if gval != -1:
                                r0, c0 = samples_px[gval]
                                dy = row - r0
                                dx = col - c0
                                dist = math.hypot(dx, dy)
                                if dist < r:
                                    ok = False
                                    break
                        if not ok:
                            break

                    if ok:
                        samples.append(p)
                        samples_px.append((row, col))
                        new_index = len(samples) - 1
                        active.append(new_index)
                        grid[gx, gy] = new_index
                        found = True
                        break
                else:
                    gx, gy = xy_to_grid(p)
                    ix0 = max(gx - neighbor_margin, 0)
                    ix1 = min(gx + neighbor_margin, gw - 1)
                    iy0 = max(gy - neighbor_margin, 0)
                    iy1 = min(gy + neighbor_margin, gh - 1)

                    ok = True
                    for ix in range(ix0, ix1 + 1):
                        for iy in range(iy0, iy1 + 1):
                            gval = grid[ix, iy].item()
                            if gval != -1:
                                d = torch.norm(p - samples[gval]).item()
                                if d < r:
                                    ok = False
                                    break
                        if not ok:
                            break

                    if ok:
                        samples.append(p)
                        new_index = len(samples) - 1
                        active.append(new_index)
                        grid[gx, gy] = new_index
                        found = True
                        break

            if not found:
                active.pop(aidx)

        # 출력 이미지 생성 (H x W)
        targetPSF = torch.zeros((H, W), device=DEVICE, dtype=torch.uint8)
        if pixel_strict:
            rc = torch.tensor(samples_px, dtype=torch.int64)
            if rc.numel() > 0:
                rc = torch.unique(rc, dim=0)
                targetPSF[rc[:, 0], rc[:, 1]] = 255
        else:
            S = torch.stack(samples)
            rows = torch.clamp(S[:, 1].round().long(), 0, H - 1)
            cols = torch.clamp(S[:, 0].round().long(), 0, W - 1)
            rc = torch.stack([rows, cols], dim=1).unique(dim=0)
            targetPSF[rc[:, 0], rc[:, 1]] = 255

        out = targetPSF.detach().cpu().numpy()
        print("Done: {:.2f}s\n".format(time() - self.start_time))
        return out


def knn_min_distances(coords, batch=20000, device="cpu"):
    """
    coords: (N,2) tensor (float or long), 같은 좌표계에서 유클리드 거리.
    1-NN(자기 자신 제외) 거리를 근사/정확 계산.
    """
    X = coords.to(device).float()
    N = X.shape[0]
    mins = []

    # 타일링으로 메모리 절약
    for i in range(0, N, batch):
        Xb = X[i : i + batch]  # (B,2)
        # 전체와의 거리
        d2 = torch.cdist(Xb, X)  # (B,N)
        # 자기 자신 제거
        idx = torch.arange(i, min(i + batch, N), device=device)
        d2[torch.arange(len(idx)), idx] = float("inf")
        mins.append(d2.min(dim=1).values.cpu())
    mins = torch.cat(mins, dim=0)
    return mins


def verify_cutoff(samples_float_xy=None, image_rc=None, r=50.0, device="cpu"):
    """
    samples_float_xy: (N,2) float tensor, (x,y)
    image_rc: (N,2) long tensor, (row, col)
    r: 기대 컷오프 in pixels
    """
    if samples_float_xy is not None:
        rows_f = torch.clamp(samples_float_xy[:, 1].round().long(), 0, 10**9)
        cols_f = torch.clamp(samples_float_xy[:, 0].round().long(), 0, 10**9)
        Df = knn_min_distances(samples_float_xy, device=device)
        print(
            f"[연속 좌표] min={Df.min().item():.2f}, mean={Df.mean().item():.2f}, "
            f"pct(<r)={(Df<r).float().mean().item()*100:.2f}%"
        )

    if image_rc is not None:
        # 중복 제거
        rc = torch.unique(image_rc, dim=0)
        # (row,col) → (x,y)로 바꿔도 되지만, row= y, col= x이므로 그대로 유클리드
        Dp = knn_min_distances(rc.float(), device=device)
        print(
            f"[픽셀 좌표] min={Dp.min().item():.2f}, mean={Dp.mean().item():.2f}, "
            f"pct(<r)={(Dp<r).float().mean().item()*100:.2f}%"
        )


try:
    import pandas as pd
except ImportError:
    pd = None  # pandas가 없어도 스크립트가 로드될 수 있도록 처리


# -------------------------------
# Save parameter set to Excel/CSV
# -------------------------------
def save_parameters(params, output_dir, filename):
    """
    파라미터 리스트를 Excel 또는 CSV로 저장하는 헬퍼 함수
    """
    excel_path = os.path.join(output_dir, "pattern_settings.xlsx")
    csv_path = os.path.join(output_dir, "pattern_settings.csv")

    try:
        if pd is None:
            raise ImportError("pandas not found")  # pandas가 없으면 CSV로 바로 이동

        df = pd.DataFrame(params, columns=["name", "value", "unit", "description"])
        df["value"] = df["value"].apply(
            lambda v: f"{v:.9g}" if isinstance(v, float) else v
        )
        df.to_excel(excel_path, index=False)
        print(f"[OK] Params saved to Excel: {excel_path}")

    except Exception as e:
        # fallback: CSV 저장
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "value", "unit", "description"])
            for row in params:
                writer.writerow(row)
        print(f"[OK] pandas 미설치/에러로 CSV로 저장했습니다: {csv_path}\nReason: {e}")


#####################################################################################
##################################                 ##################################
##################################   Main Function ##################################
##################################                 ##################################
#####################################################################################


def run_generation(
    out_W,
    out_H,
    dot_distance_multiplier,
    pixel_size_m=2.7e-6,
    height_max_um=10,
    obj_dist_mm=6.5,
    img_dist_mm=1.1,
    n=1.53,
):
    """
    지정된 파라미터로 랜덤 닷 패턴을 생성하고 저장합니다.

    :param out_W: 출력 이미지 너비 (pixels)
    :param out_H: 출력 이미지 높이 (pixels)
    :param dot_distance_multiplier: 'd'에 곱해줄 Dot2Dot 거리 계수 (예: 1.3)
    :param pixel_size_m: 픽셀 크기 (meters)
    :param height_max_um: 최대 높이 (micrometers)
    :param obj_dist_mm: 물체 거리 (mm)
    :param img_dist_mm: 이미지 거리 (mm)
    :param n: 굴절률 (Refractive_Index)
    """
    print(f"\n--- Starting Generation ---")
    print(f"Params: W={out_W}, H={out_H}, Multiplier={dot_distance_multiplier}")

    # --- 1. 파라미터 계산 ---
    Pixel_Size = pixel_size_m
    MaskSize_um = 5400  # um 단위 (이 값은 M 계산에만 쓰였으므로, out_W/out_H를 직접 받아와서 큰 의미는 없을 수 있습니다)
    MaskSize = MaskSize_um * 1e-6  # m 단위
    # M = int(np.floor(MaskSize / Pixel_Size)) # M 대신 out_W, out_H를 직접 사용

    HeightProfile_Max_um = height_max_um
    H_max = HeightProfile_Max_um * 1e-6
    ObjectDistance_mm = obj_dist_mm
    ImageDistance_mm = img_dist_mm
    Refractive_Index = n

    EFL_UnitLens_mm = 1 / (1 / ObjectDistance_mm + 1 / ImageDistance_mm)
    Radius_of_Curvature_UnitLens_mm = EFL_UnitLens_mm * (Refractive_Index - 1)
    R = Radius_of_Curvature_UnitLens_mm * 1e-3
    d = np.sqrt(R**2 - (R - H_max) ** 2)  # 반지름

    Dot2Dot_Distance_um = d * 1e6 * dot_distance_multiplier
    percent_aperture_values = [1]  # List of aperture percentages

    print(f"Mask size : {MaskSize_um/1000} mm")
    print(f"Object distance : {ObjectDistance_mm} mm")
    print(f"Image distance : {ImageDistance_mm} mm")
    print(f"EFL : {EFL_UnitLens_mm} mm")
    print(f"Radius of curvature : {Radius_of_Curvature_UnitLens_mm} mm")
    print(f"Radius of circle : {d*1000} mm")
    print(
        f"Dot to dot distance : {Dot2Dot_Distance_um:.2f} um\n"
    )  # 소수점 2자리로 포맷팅

    # --- 2. 닷 생성 ---# --- 2. 닷 생성 ---
    RDG = RandomDotsGenerator(
        target_size=(out_W, out_H),  # (예: 2560, 1600)
        avg_distance_um=Dot2Dot_Distance_um,
        pixel_size=Pixel_Size,
        visualize=False,
    )
    # Pattern_Poisson은 (H, W) 크기 (예: 1600, 2560)의 직사각형 배열
    Pattern_Poisson = RDG.PoissonDiskRandomDots()

    print("Poisson Disk Sampling : done")
    print(f"Target Dots : {RDG.numDots}ea")

    # rc 좌표는 원본 (1600, 2560) 기준으로 계산 (정확함)
    rc = torch.nonzero(
        torch.tensor(Pattern_Poisson) > 0, as_tuple=False
    )  # (N,2), row,col
    print(f"Actual Dots : {rc.shape[0]}ea\n")

    S = None
    min_dist_pixels = Dot2Dot_Distance_um / (Pixel_Size * 1e6)
    # verify_cutoff도 원본 (1600, 2560) 기준으로 수행 (정확함)
    verify_cutoff(
        samples_float_xy=S,
        image_rc=rc,
        r=min_dist_pixels,
        device=DEVICE,
    )

    # --- [신규] 2.5. 정사각형으로 제로 패딩 ---
    gen_H, gen_W = Pattern_Poisson.shape
    max_side = max(gen_W, gen_H)  # 예: max(2560, 1600) -> 2560

    out_final_W = max_side
    out_final_H = max_side

    Pattern_Square = Pattern_Poisson  # 기본값은 원본

    if gen_H != gen_W:
        print(
            f"Padding rectangular pattern ({gen_H}x{gen_W}) to square ({max_side}x{max_side})..."
        )
        Pattern_Square = np.zeros((max_side, max_side), dtype=np.uint8)

        # 중앙 정렬을 위한 패딩 계산
        pad_H_top = (max_side - gen_H) // 2
        pad_H_bottom = max_side - gen_H - pad_H_top
        pad_W_left = (max_side - gen_W) // 2
        pad_W_right = max_side - gen_W - pad_W_left

        # 중앙에 붙여넣기
        Pattern_Square[
            pad_H_top : max_side - pad_H_bottom, pad_W_left : max_side - pad_W_right
        ] = Pattern_Poisson

        print(
            f"Padding complete. Top:{pad_H_top}, Bottom:{pad_H_bottom}, Left:{pad_W_left}, Right:{pad_W_right}"
        )

    # --- 3. 파일 저장 ---
    now = datetime.datetime.now()
    filename_time = now.strftime("%Y%m%d_%H%M%S")

    # [수정됨] 폴더 이름에 최종 정사각형 크기(max_side) 사용
    Base_dir = f"./Outputs/random_dots_M_{out_final_W}x{out_final_H}_{RDG.numDots}dots_mult_{dot_distance_multiplier:.2f}_{filename_time}"
    Poisson_dir = os.path.join(Base_dir, "PoissonDiskRandomDots")
    os.makedirs(Poisson_dir, exist_ok=True)

    # [수정됨] 파일 이름에 최종 정사각형 크기(max_side) 사용
    filename_base = f"PoissonDiskRandomDots_M_{out_final_W}x{out_final_H}_avg_dist_{Dot2Dot_Distance_um:.0f}um_mult_{dot_distance_multiplier:.2f}"
    filename_png = f"{filename_time}_{filename_base}.png"

    # [수정됨] Pattern_Square (패딩된 정사각형 이미지)를 저장
    cv2.imwrite(os.path.join(Poisson_dir, filename_png), Pattern_Square)
    print(f"Pattern saved in {Base_dir}")

    # --- 4. 파라미터 저장 ---
    params = [
        ("Pixel_Size", Pixel_Size, "m", "Pixel size (image plane)"),
        ("MaskSize_um", MaskSize_um, "um", "Mask size (micrometers)"),
        ("MaskSize", MaskSize, "m", "Mask size (meters)"),
        # [신규] 원본 생성 크기 (직사각형)
        ("Gen_M_width", out_W, "-", "Generation size width (pixels)"),
        ("Gen_M_height", out_H, "-", "Generation size height (pixels)"),
        # [수정됨] 최종 출력 크기 (정사각형)
        ("M_width", out_final_W, "-", "Output size width (pixels)"),
        ("M_height", out_final_H, "-", "Output size height (pixels)"),
        ("HeightProfile_Max_um", HeightProfile_Max_um, "um", "Max height of mask"),
        ("H_max", H_max, "m", "Max height of mask (meters)"),
        ("ObjectDistance_mm", ObjectDistance_mm, "mm", "Object distance"),
        ("ImageDistance_mm", ImageDistance_mm, "mm", "Image distance"),
        ("Refractive_Index", Refractive_Index, "-", "Refractive index"),
        ("EFL_UnitLens_mm", EFL_UnitLens_mm, "mm", "Effective focal length"),
        (
            "Radius_of_Curvature_UnitLens_mm",
            Radius_of_Curvature_UnitLens_mm,
            "mm",
            "Radius of curvature",
        ),
        ("R", R, "m", "Radius of curvature (meters)"),
        ("d", d, "m", "Circle radius from sag"),
        (
            "dot_distance_multiplier",
            dot_distance_multiplier,
            "-",
            "Multiplier for d*1e6",
        ),
        (
            "Dot2Dot_Distance_um",
            Dot2Dot_Distance_um,
            "um",
            "Target Poisson min distance",
        ),
        (
            "min_dist_pixels",
            min_dist_pixels,
            "pixels",
            "Target min distance (in pixels)",
        ),
        (
            "percent_aperture_values",
            str(percent_aperture_values),
            "-",
            "Aperture percent list",
        ),
        # [수정됨] Target Dots 개수 설명 변경
        (
            "Target_Dots_Count",
            int(RDG.numDots),
            "ea",
            "Number of target dots (in Gen_M area)",
        ),
        (
            "Actual_Dots_Count",
            int(rc.shape[0]),
            "ea",
            "Number of actual generated dots",
        ),
        ("Output_Image_File", filename_png, "-", "Saved image filename"),
        ("Output_Folder", os.path.abspath(Poisson_dir), "-", "Output directory"),
        ("Saved_At", now.isoformat(timespec="seconds"), "-", "Save timestamp (local)"),
    ]

    save_parameters(params, Poisson_dir, filename_base)
    print(f"--- Generation Complete ---\n")


if __name__ == "__main__":
    # 이 파일을 직접 실행할 때의 기본 동작

    # --- 기본 파라미터 ---
    DEFAULT_W = 2560
    DEFAULT_H = 1600
    DEFAULT_MULTIPLIER = 2.0

    print("Running single generation with default parameters...")
    run_generation(
        out_W=DEFAULT_W, out_H=DEFAULT_H, dot_distance_multiplier=DEFAULT_MULTIPLIER
    )
