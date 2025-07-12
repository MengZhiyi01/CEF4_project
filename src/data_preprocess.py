import os
import pandas as pd
import numpy as np
import glob
import sys
sys.path.append("..")  # 添加上级目录

from config.contract_margin import contract_margin_ratio
from config.contract_multiplier import contract_multipliers

import numpy as np
import pandas as pd
from datetime import time

def process_all_csv(input_folder, output_folder):
    """
    对 input_folder 中所有 CSV 文件执行数据处理操作，并将结果保存到 output_folder 中。
    处理包括：
        - 从文件名提取品种代码并添加到列中
        - 将 'Unnamed: 0' 转换为 datetime 类型并重命名为 'datetime'
        - 只保留6月30日及之前的数据
        - 删除原始列
        - 计算涨跌幅 change
        - 计算 VWAP
        - 重命名 pre_close 为 preclose
        - 简化输出文件名
    """
    # 获取所有 csv 文件路径
    csv_files = glob(os.path.join(input_folder, "*.csv"))

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for file_path in csv_files:

        # 从文件名提取品种代码 (如 A9999.XDCE)
        filename = os.path.basename(file_path)
        code = filename.split('_')[0]
            
        # 读取 CSV
        df = pd.read_csv(file_path)

        # 添加品种代码列
        df['code'] = code

        # 转换并重命名时间列
        df['datetime'] = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop(columns=['Unnamed: 0'])

        # 筛选6月30日及之前的数据
        df = df[df['datetime'].dt.month > 6]  # 修改为 <=6 以保留6月及之前的数据

        # 计算涨跌幅和 VWAP
        df['change'] = (df['close'] - df['open']) / df['open'] * 100
        df['vwap'] = df['money'] / df['volume'] / contract_multipliers[code]

        # 重命名 pre_close
        df['preclose'] = df['close'].shift(1)  # 假设 pre_close 是前一行的 close
        if 'pre_close' in df.columns:
            df = df.drop(columns=['pre_close'])

        # 保持列顺序，使 datetime 和 code 在前
        cols = ['datetime', 'code'] + [col for col in df.columns if col not in ['datetime', 'code']]
        df = df[cols]
        df['vwap'] = df['vwap'].fillna(df['close'])

        is_out_of_range = (df['vwap'] > df['high']) | (df['vwap'] < df['low'])
        # df['vwap_original'] = df['vwap']  # 备份原始值

        # 3. 处理异常值（增加边界值检查）
        df.loc[is_out_of_range, 'vwap'] = np.nan

        # 4. 插值处理（增加method参数说明）
        df['vwap'] = df['vwap'].interpolate(
            method='linear',  # 时间序列推荐'linear'或'time'
            limit_direction='both',  # 双向插值
            limit_area='inside'      # 只填充被有效值包围的NaN
        )

        # 5. 处理残余NaN（首尾异常值）
        df['vwap'] = df['vwap'].fillna(
            (df['high'] + df['low']) / 2  # 用高低均价填充无法插值的NaN
        )

        # 检查是否有越界的 VWAP 值
        mask_out_of_range = (df['vwap'] < df['low']) | (df['vwap'] > df['high'])
        # 2. 打印统计信息
        count = mask_out_of_range.sum()
        total = len(df)
        percentage = 100 * count / total if total > 0 else 0

        print(f"共发现 {count} 个 vwap 超出 [low, high] 区间的样本，占比 {percentage:.2f}%")
        if mask_out_of_range.any():
            print("提示：发现 vwap 超出 [low, high] 区间，已执行 clip 处理。")
            df['vwap'] = df['vwap'].clip(lower=df['low'], upper=df['high'])


        missing_money = df['money'].isna().sum()
        missing_volume = df['volume'].isna().sum()
        missing_vwap = df['vwap'].isna().sum()
            
        print(f"\nMissing values in {code}:")
        print(f"- money: {missing_money} missing")
        print(f"- volume: {missing_volume} missing")
        print(f"- vwap: {missing_vwap} missing")
            
        # 如果有缺失值，输出前5行缺失值的位置
        if missing_money > 0 or missing_volume > 0 or missing_vwap > 0:
            print("\nSample rows with missing values:")
            missing_rows = df[df[['money', 'volume', 'vwap']].isna().any(axis=1)]
            print(missing_rows.head())

        # 构建输出路径，只使用代码作为文件名
        output_path = os.path.join(output_folder, f"{code}.csv")

        # 保存为 utf-8-sig 编码
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nProcessed and saved: {code}.csv")


def process_tick_data(df: pd.DataFrame, code: str, save_path: str):
    """
    处理逐笔数据，生成1分钟K线数据，并保存为CSV文件。
    
    :param df: 输入的逐笔数据 DataFrame
    :param code: 股票代码
    :param save_path: 保存路径
    """
# 计算金额
    df['Amount'] = df['Price'] * df['Volume']

    # 新增列
    df['abvol']   = np.where(df['Side'] == 0, df['Volume'], 0)
    df['abamount'] = np.where(df['Side'] == 0, df['Amount'], 0)

    df['asvol']   = np.where(df['Side'] == 1, df['Volume'], 0)
    df['asamount'] = np.where(df['Side'] == 1, df['Amount'], 0)
    df.loc[df['DealTime'] <= 93000000, 'DealTime'] = 93000001
    df.loc[df['DealTime'] > 150000000, 'DealTime'] = 150000000
    df.loc[(df['DealTime'] > 113000000) & (df['DealTime'] < 130000000), 'DealTime'] = 113000000
    df.loc[(df['DealTime'] == 130000000), 'DealTime'] = 130000001
        # 构造 datetime 字段
    df['datetime'] = pd.to_datetime(
            df['TradingDay'].astype(str) + df['DealTime'].astype(str).str.zfill(9),
            format='%Y%m%d%H%M%S%f'
        )

        # 删除 Price 或 amount 为 0 的行
    df = df[df['Price'] > 0].copy()
    df = df[df['Amount'] > 0].copy()

    # 生成 1 分钟 K 线
    df_1min = (
            df.set_index('datetime')
            .groupby('SecuCode')
            .resample('1T', closed='right', label='right')
            .agg({
                'Price': ['first', 'last', 'max', 'min'],
                'Volume': 'sum',
                'Amount': 'sum',
                'abvol': 'sum',
                'abamount': 'sum',
                'asvol': 'sum',
                'asamount': 'sum'
            })
        )
    df_1min

    # 展平列名（'Price', 'first') → 'Price_first'）
    df_1min.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_1min.columns]

    # 重置索引（如果你想把 SecuCode 和 datetime 变成普通列）
    df_1min = df_1min.reset_index()
    df_1min['date'] = df_1min['datetime'].dt.date
    df_1min['time'] = df_1min['datetime'].dt.time
    df_1min.columns = ['stock_code','datetime','open', 'close', 'high', 'low', 'volume', 'amount','abvol', 'abamount', 'asvol', 'asamount', 'date', 'time']
    

    # 读取数据（如果你已经有 df_1min，跳过此步）
    # df_1min = pd.read_csv("your_file.csv")

    # 确保 datetime 列是时间格式
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])

    # 创建一个新列表示时间部分（用于筛选）
    df_1min['time_only'] = df_1min['datetime'].dt.time

    # 找到所有 14:57:00 的收盘价
    close_1457 = (
        df_1min[df_1min['time_only'] == pd.to_datetime("14:57:00").time()]
        .set_index(['stock_code', 'date'])['close']
    )

    # 定义需要修改的目标时间
    target_times = ["14:58:00", "14:59:00"]

    # 遍历目标时间进行赋值
    for t in target_times:
        time_obj = pd.to_datetime(t).time()
        mask = df_1min['time_only'] == time_obj
        key = list(zip(df_1min.loc[mask, 'stock_code'], df_1min.loc[mask, 'date']))
        replacement_values = [close_1457.get(k, float('nan')) for k in key]

        for col in ['open', 'close', 'high', 'low']:
            df_1min.loc[mask, col] = replacement_values

    # 如果不再需要 time_only 列可以删除
    df_1min.drop(columns=['time_only'], inplace=True)
    

    # 筛选出有效交易日（volume > 0）
    daily_volume = df_1min[df_1min['volume'].notna()].groupby('date')['volume'].sum()
    valid_dates = daily_volume[daily_volume > 0].index
    df_1min = df_1min[df_1min['date'].isin(valid_dates)].copy()

    # 保留交易时间段内的数据
    df_1min['time'] = df_1min['datetime'].dt.time
    morning_start = time(9, 30)
    morning_end = time(11, 30)
    afternoon_start = time(13, 0)
    afternoon_end = time(15, 0)
    df_filtered = df_1min[
            ((df_1min['time'] > morning_start) & (df_1min['time'] <= morning_end)) |
            ((df_1min['time'] > afternoon_start) & (df_1min['time'] <= afternoon_end))
        ].copy()
    # 添加涨跌幅、VWAP 等字段
    df_filtered['ret'] = ((df_filtered['close'] - df_filtered['open']) / df_filtered['open']).fillna(0) * 100
    df_filtered['vwap'] = df_filtered['amount'] / df_filtered['volume']
    df_filtered['stock_code'] = df_filtered['stock_code'].astype(str)
    # 保存为 CSV 文件
    df_filtered.to_csv(f'{save_path}/{code}.csv', index=False, encoding='utf-8-sig')
    # print(f"保存成功：{save_path}/{code}.csv")

import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_stock(code, folder_path, save_path):
    """处理单个股票文件的函数"""
    file_path = os.path.join(folder_path, f"{code}.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            process_tick_data(df, code, save_path=save_path)
            return code, True  # 返回股票代码和成功状态
        except Exception as e:
            print(f"处理 {code} 时出错: {str(e)}")
            return code, False
    else:
        print(f"文件不存在: {file_path}")
        return code, False

def parallel_process_stocks(zz500_stocks, folder_path, save_path, max_workers=24):
    """并行处理股票数据"""
    total = len(zz500_stocks['code'])
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_stock, code, folder_path, save_path): code
            for code in zz500_stocks['code']
        }
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=total, desc="并行处理股票数据"):
            code, status = future.result()
            if status:
                success_count += 1
    
    print(f"处理完成！成功处理 {success_count}/{total} 支股票")

if __name__ == '__main__':
    # 示例用法：
    input_folder = r"C:\baidunetdiskdownload\解压工具(必须)\output\主力连续_1m_05-24\2024主力连续_1min"
    output_folder = r"C:\Users\36119\Desktop\CEF4\9999_1m_0708v3"
    process_all_csv(input_folder, output_folder)

    # 使用示例
    folder_path = "E:\\level2_data\\deal\\2024\\20241201_20250101\\"
    save_path = 'C:\\Users\\36119\\Desktop\\CEF4\\zz500_2412_1T'

    parallel_process_stocks(pd.DataFrame({'code':['300114']}), folder_path, save_path)