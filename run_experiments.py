# gen_random_dot_pixel_new.py 파일에서 run_generation 함수를 가져옵니다.
from gen_random_dot_pixel_new import run_generation
import time

# # ---------------------------------
# # 목표 1: 작은 규모로 테스트 실행
# # ---------------------------------
# print("=================================")
# print("===   Running Small-Scale Test  ===")
# print("=================================")

# start_test = time.time()
# run_generation(
#     out_W=300,  # 테스트용 작은 너비
#     out_H=200,  # 테스트용 작은 높이
#     dot_distance_multiplier=1.3,  # 테스트용 계수
#     # 다른 파라미터(pixel_size 등)는 기본값을 사용합니다.
# )
# end_test = time.time()
# print(f"[OK] Small-scale test finished in {end_test - start_test:.2f} seconds.")


# ---------------------------------
# 목표 2: 파라미터 스윕 실행
# ---------------------------------
print("\n===================================")
print("=== Running Parameter Sweep ===")
print("===================================")

# ---- 실험 설정 ----
# 실제 사용할 이미지 크기
full_W = 2560
full_H = 1600

# 변경해 볼 Dot2Dot_Distance_um 계수 리스트
# multiplier_list = [1.0, 1.3, 1.5, 2.0]
multiplier_list = [2.0]
# ---------------------

print(f"Target size: {full_W}x{full_H}")
print(f"Multipliers to test: {multiplier_list}")

start_sweep = time.time()
for i, mult in enumerate(multiplier_list):
    print(f"\n--- Running Sweep {i+1}/{len(multiplier_list)} (Multiplier: {mult}) ---")
    run_generation(out_W=full_W, out_H=full_H, dot_distance_multiplier=mult)
end_sweep = time.time()

print("===================================")
print("===     Parameter Sweep Done    ===")
print(f"Total sweep time: {(end_sweep - start_sweep) / 60:.2f} minutes.")
