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
    confirm_times: int = 1  # 新增参数：连续信号确认次数
) -> pd.DataFrame:
    trade_records = []

    for contract, contract_df in merged_df.groupby('contract'):
        contract_df = contract_df.sort_values('datetime').reset_index(drop=True)
        contract_df.set_index('datetime', inplace=True)

        position = None
        long_confirm_count = 0
        short_confirm_count = 0

        for curr_time, row in contract_df.iterrows():
            if position is None:
                # 累加确认次数
                if row['pred'] > long_threshold:
                    long_confirm_count += 1
                    short_confirm_count = 0
                elif row['pred'] < short_threshold:
                    short_confirm_count += 1
                    long_confirm_count = 0
                else:
                    long_confirm_count = 0
                    short_confirm_count = 0

                # 开仓条件：连续满足阈值
                if long_confirm_count >= confirm_times:
                    position = 'long'
                elif short_confirm_count >= confirm_times:
                    position = 'short'
                else:
                    continue  # 没达到确认次数，不开仓

                entry_time = curr_time + pd.Timedelta(minutes=1)

                if entry_time in contract_df.index:
                    entry_price = contract_df.loc[entry_time, 'vwap']
                else:
                    future_prices = contract_df.loc[contract_df.index >= entry_time]
                    if future_prices.empty:
                        position = None
                        continue  # 或 break，跳过这次交易
                    entry_price = future_prices.iloc[0]['vwap']
                    entry_time = future_prices.index[0]  # 更新真实的 entry_time
                exit_time = entry_time + pd.Timedelta(minutes=holding_minutes-1)

                margin_ratio = row['Margin_ratio']
                multiplier = row['Multiplier']

                position_size = int(position_ratio * account / (margin_ratio * entry_price * multiplier))
                if position_size == 0:
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

                    position = None
                    long_confirm_count = 0
                    short_confirm_count = 0  # 重置确认计数

    return pd.DataFrame(trade_records)
