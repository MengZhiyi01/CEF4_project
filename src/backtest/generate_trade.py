import pandas as pd

def generate_trade_records(
    merged_df: pd.DataFrame,
    long_threshold: float = 0.0008,
    short_threshold: float = -0.0008,
    fee_rate: float = 1e-4,
    risk_control: bool = True,
    account: float = 2_000_000,
    position_ratio: float = 0.1,
    holding_minutes: int = 5,
) -> pd.DataFrame:
    """
    根据多空信号和资金控制，生成交易记录（开仓和平仓）回测结果。

    参数:
        merged_df: 输入包含预测信号和行情的DataFrame，必须包含列：
                   ['datetime', 'contract', 'pred', 'vwap', 'Multiplier', 'Margin_ratio']
        long_threshold: 开多仓阈值
        short_threshold: 开空仓阈值
        fee_rate: 手续费率（开平仓均计算）
        risk_control: 是否启用爆仓风险控制（亏损不超过保证金）
        account: 初始账户资金
        position_ratio: 单笔交易占用资金比例
        holding_minutes: 持仓时间（分钟）

    返回:
        pd.DataFrame: 包含所有交易记录，列有合同、方向、开平仓时间价格、收益等
    """

    trade_records = []

    for contract, contract_df in merged_df.groupby('contract'):
        contract_df = contract_df.sort_values('datetime').reset_index(drop=True)
        contract_df.set_index('datetime', inplace=True)

        position = None

        for curr_time, row in contract_df.iterrows():
            if position is None:
                if row['pred'] > long_threshold:
                    position = 'long'
                elif row['pred'] < short_threshold:
                    position = 'short'
                else:
                    continue  # 无信号不开仓

                entry_time = curr_time + pd.Timedelta(minutes=1)
                exit_time = curr_time + pd.Timedelta(minutes=holding_minutes)
                entry_price = row['vwap']
                margin_ratio = row['Margin_ratio']
                multiplier = row['Multiplier']

                # 计算单笔持仓手数
                position_size = int(position_ratio * account / (margin_ratio * entry_price * multiplier))
                if position_size == 0:
                    # 保证金不足，不开仓
                    position = None
                    continue

            else:
                # 持仓中，等待平仓时机
                if curr_time >= exit_time:
                    if exit_time in contract_df.index:
                        exit_price = contract_df.loc[exit_time, 'vwap']
                    else:
                        future_prices = contract_df.loc[contract_df.index >= exit_time]
                        if len(future_prices) == 0:
                            break
                        exit_price = future_prices.iloc[0]['vwap']
                        exit_time = future_prices.index[0]

                    direction_factor = 1 if position == 'long' else -1
                    ret = (exit_price - entry_price) / entry_price * direction_factor
                    gross_pnl = ret * position_size * multiplier * entry_price
                    fee = (entry_price + exit_price) * position_size * multiplier * fee_rate
                    net_pnl = gross_pnl - fee
                    trade_value = position_size * multiplier * entry_price
                    margin_required = trade_value * margin_ratio

                    # 爆仓风险控制
                    if gross_pnl < -margin_required and risk_control:
                        gross_pnl = -margin_required
                        net_pnl = gross_pnl - fee

                    trade_records.append({
                        'contract': contract,
                        'direction': position,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'return': ret,
                        'gross_pnl': gross_pnl,
                        'fee': fee,
                        'net_pnl': net_pnl,
                        'margin': margin_required,
                        'position_size': position_size
                    })

                    position = None  # 清仓

    return pd.DataFrame(trade_records)