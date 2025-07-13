import os
import subprocess  # 用于在Python中执行终端命令
from tqdm.auto import tqdm

# --- 用户需要配置的路径 ---
base_dir = os.path.dirname(os.path.abspath(__file__))
# 1. 存放您所有原始CSV文件的输入文件夹
source_csv_folder = os.path.join(base_dir, '../data/input/9999_1m_0708v3')
# 2. 用于存放所有独立Qlib数据仓库的根目录
output_base_folder = os.path.join(base_dir, '../data/output/individual_qlibs_0708v3')
# 3. 您的 dump_bin.py 脚本的路径
dump_script_path = "/opt/anaconda3/envs/py38/lib/python3.8/site-packages/qlib-main/scripts/dump_bin.py"

# --- 自动化处理脚本 ---

# 确保输出根目录存在
os.makedirs(output_base_folder, exist_ok=True)

# 获取所有CSV文件列表
try:
    csv_files = [f for f in os.listdir(source_csv_folder) if f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个CSV文件准备处理。")
except FileNotFoundError:
    print(f"错误：找不到源文件夹 {source_csv_folder}")
    csv_files = []

# 对每个CSV文件执行独立的dump操作
for csv_file in tqdm(csv_files, desc="正在为每个资产创建独立的Qlib仓库"):
    # 构建输入和输出路径
    full_csv_path = os.path.join(source_csv_folder, csv_file)
    instrument_name = os.path.splitext(csv_file)[0]  # 从文件名获取资产名
    instrument_qlib_dir = os.path.join(output_base_folder, instrument_name)  # 创建专属输出目录

    # 构建完整的 dump_bin.py 命令
    command = [
        "python",
        dump_script_path,
        "dump_all",
        "--csv_path", full_csv_path,  # 每次只处理一个CSV文件
        "--qlib_dir", instrument_qlib_dir,  # 每次都使用专属的输出目录
        "--freq", "1min",
        "--symbol_field_name", "code",  # 根据您之前的命令
        "--date_field_name", "datetime",
        "--include_fields",
        "open,close,high,low,volume,money,avg,high_limit,low_limit,paused,factor,open_interest,change,vwap,preclose"
    ]

    # 执行命令
    # print(f"\n正在执行命令: {' '.join(command)}") # 如果需要调试，可以取消这行注释
    subprocess.run(command, check=True, capture_output=True, text=True)

print("\n所有资产的独立Qlib数据仓库已成功创建！")
