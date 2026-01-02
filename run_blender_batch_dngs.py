import os
import subprocess
import json
import statistics

# ==================== 配置区域 ====================

# 1. 数据集根目录 (请根据实际情况修改)
# 注意：如果你是在 Windows 上跑，但数据在 WSL 里，请用 r"\\wsl.localhost\..."
# 如果数据就在上级目录，保持默认即可
DATASET_ROOT = r"../small-data/nerf_synthetic_dngaussian"
# 或者使用绝对路径，例如:
# DATASET_ROOT = r"\\wsl.localhost\Ubuntu-20.04-C\root\small-data\nerf_synthetic_dngaussian"

# 2. 输出根目录
OUTPUT_ROOT = r"output/blender"

# 3. 场景列表
# SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
SCENES = ["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

# 4. 固定参数 (除了 -s 和 --model_path 之外的所有参数)
COMMON_ARGS = [
    "-r", "2",
    "--eval",
    "--rand_pcd",
    "--iterations", "7000",
    "--lambda_dssim", "0.2",
    "--densify_grad_threshold", "0.0002",
    "--prune_threshold", "0.005",
    "--densify_until_iter", "5000",
    "--percent_dense", "0.01",
    "--densify_from_iter", "500",
    "--position_lr_init", "0.00016",
    "--position_lr_final", "0.0000016",
    "--position_lr_max_steps", "7000",
    "--position_lr_start", "0",
    "--test_iterations", "7000",
    "--save_iterations", "7000",
    "--hard_depth_start", "99999",
    "--error_tolerance", "0.2",
    "--scaling_lr", "0.005",
    "--shape_pena", "0.000",
    "--opa_pena", "0.000",
    "--scale_pena", "0.000",
    "--use_SH"
]


# =================================================

def run_all_scenes():
    print(f"Dataset Root: {os.path.abspath(DATASET_ROOT)}")
    print(f"Output Root:  {os.path.abspath(OUTPUT_ROOT)}\n")

    for scene in SCENES:
        print(f"--- Starting Scene: {scene} ---")

        # 构建路径
        source_path = os.path.join(DATASET_ROOT, scene)
        model_path = os.path.join(OUTPUT_ROOT, scene)

        # Windows下为了安全，转换一下路径分隔符
        source_path = os.path.normpath(source_path)
        model_path = os.path.normpath(model_path)

        # 检查数据是否存在
        if not os.path.exists(source_path):
            print(f"[Warning] Path does not exist, skipping: {source_path}")
            continue

        # 组装命令
        # 对应: python train_blender.py -s ... --model_path ... [COMMON_ARGS]
        cmd = ["python", "train_blender.py", "-s", source_path, "--model_path", model_path] + COMMON_ARGS

        try:
            # 运行命令
            subprocess.run(cmd, check=True)
            print(f"--- Finished Scene: {scene} ---\n")
        except subprocess.CalledProcessError:
            print(f"!!! Error running scene: {scene} !!!\n")
        except KeyboardInterrupt:
            print("\nAborted by user.")
            return

    print("Training loop finished. Calculating metrics...\n")
    calc_metrics()


def calc_metrics():
    """
    遍历输出目录，读取 results.json 并计算平均值
    """
    metrics_data = {"PSNR": [], "SSIM": [], "LPIPS": []}

    print(f"{'Scene':<12} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8}")
    print("-" * 46)

    for scene in SCENES:
        # JSON 文件的典型路径: output/blender/chair/results.json
        json_path = os.path.join(OUTPUT_ROOT, scene, "results.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # 这里的 key 通常是 'ours_7000' 或者 'ours_30000'，或者是 'test'
                    # 我们尝试获取第一个 key 的内容 (通常只有一个 key)
                    first_key = list(data.keys())[0]
                    res = data[first_key]

                    psnr = res.get('PSNR', 0)
                    ssim = res.get('SSIM', 0)
                    lpips = res.get('LPIPS', 0)

                    metrics_data["PSNR"].append(psnr)
                    metrics_data["SSIM"].append(ssim)
                    metrics_data["LPIPS"].append(lpips)

                    print(f"{scene:<12} | {psnr:.4f}   | {ssim:.4f}   | {lpips:.4f}")
            except Exception as e:
                print(f"{scene:<12} | Error reading JSON: {e}")
        else:
            print(f"{scene:<12} | No results.json found")

    print("-" * 46)

    # 计算并打印平均值
    if metrics_data["PSNR"]:
        avg_psnr = statistics.mean(metrics_data["PSNR"])
        avg_ssim = statistics.mean(metrics_data["SSIM"])
        avg_lpips = statistics.mean(metrics_data["LPIPS"])

        print(f"{'AVERAGE':<12} | {avg_psnr:.4f}   | {avg_ssim:.4f}   | {avg_lpips:.4f}")
    else:
        print("No metrics data collected.")


if __name__ == "__main__":
    run_all_scenes()