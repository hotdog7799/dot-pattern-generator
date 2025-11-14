# gen_random_dot_pixel_new.py 파일에서 run_generation 함수를 가져옵니다.
import yaml
from gen_random_dot_pixel_new import run_generation
import time

# --- 1. 설정 파일 로드 ---
config_path = "config.yaml"
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Successfully loaded config from {config_path}")
except Exception as e:
    print(f"Error loading {config_path}: {e}")
    sys.exit(1)  # 설정 파일 없으면 종료

# --- 2. 설정 값 불러오기 ---
exp_config = config.get("experiment", {})
gen_params = config.get("generation_parameters", {})  # 딕셔너리 전체를 가져옴

# 실험 설정
full_W = exp_config.get("image_width", 2560)
full_H = exp_config.get("image_height", 1600)
multiplier_list = exp_config.get("multipliers", [2.0])

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

print(f"Target size: {full_W}x{full_H}")
print(f"Multipliers to test: {multiplier_list}")

start_sweep = time.time()
for i, mult in enumerate(multiplier_list):
    print(f"\n--- Running Sweep {i+1}/{len(multiplier_list)} (Multiplier: {mult}) ---")

    # --- 3. run_generation 호출 ---
    # YAML에서 읽어온 기본 파라미터와 실험 파라미터를 합쳐서 전달
    run_params = {
        **gen_params,  # YAML의 'generation_parameters'가 기본값으로 사용됨
        "out_W": full_W,
        "out_H": full_H,
        "dot_distance_multiplier": mult,
        # 'n' 인자 이름이 다르다면 여기서 매핑: 'n': gen_params.get('refractive_index', 1.53)
    }

    # **run_params는 딕셔너리를 함수의 키워드 인자로 자동 매핑해줍니다.
    run_generation(**run_params)

end_sweep = time.time()

print("===================================")
print("===     Parameter Sweep Done    ===")
print(f"Total sweep time: {(end_sweep - start_sweep) / 60:.2f} minutes.")
