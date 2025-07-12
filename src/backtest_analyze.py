import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_trade_performance(trade_records_df: pd.DataFrame, account: float = 2_000_000, trading_days_per_year: int = 252):
    """
    对交易记录进行每日净值统计、可视化及关键指标计算（年化收益、波动、夏普比率、最大回撤）。

    参数：
    - trade_records_df: 包含交易记录的 DataFrame，必须含列：
        ['entry_time', 'exit_time', 'net_pnl', 'margin']
    - account: 初始资金，默认200万
    - trading_days_per_year: 年交易天数，默认252

    返回：
    - daily_df: 包含每日累计净收益、保证金等的 DataFrame
    """
    # 转换时间格式，按平仓时间排序
    trade_records_df['exit_time'] = pd.to_datetime(trade_records_df['exit_time'])
    trade_records_df = trade_records_df.sort_values('exit_time').reset_index(drop=True)

    # 累计净收益
    trade_records_df['cum_net_pnl'] = trade_records_df['net_pnl'].cumsum()

    # 生成保证金时间序列
    margin_records = []
    for _, row in trade_records_df.iterrows():
        start = pd.to_datetime(row['entry_time'])
        end = pd.to_datetime(row['exit_time'])
        margin = row['margin']
        current_time = start
        while current_time < end:
            margin_records.append({'datetime': current_time, 'margin': margin})
            current_time += pd.Timedelta(minutes=1)
    margin_df = pd.DataFrame(margin_records)
    margin_df = margin_df.groupby('datetime')['margin'].sum().reset_index()

    # 合并净收益和保证金时间序列
    pnl_df = trade_records_df[['exit_time', 'cum_net_pnl']].copy()
    plot_df = pd.merge(pnl_df, margin_df, left_on='exit_time', right_on='datetime', how='outer')
    plot_df = plot_df.sort_values('exit_time').rename(columns={'exit_time': 'time'})
    plot_df['cum_net_pnl'] = plot_df['cum_net_pnl'].ffill().fillna(0)
    plot_df['margin'] = plot_df['margin'].fillna(0)

    # 日线数据
    plot_df['date'] = plot_df['time'].dt.floor('D')
    daily_cum_pnl = plot_df.groupby('date')['cum_net_pnl'].last()
    daily_margin = plot_df.groupby('date')['margin'].mean()
    daily_df = pd.DataFrame({'cum_net_pnl': daily_cum_pnl, 'margin': daily_margin}).reset_index()
    daily_df['daily_pnl'] = daily_df['cum_net_pnl'].diff().fillna(0)

    # 计算指标
    num_days = len(daily_df)
    total_return = daily_df['cum_net_pnl'].iloc[-1] / account
    annual_return = total_return / num_days * trading_days_per_year
    daily_volatility = (daily_df['daily_pnl'] / account).std()
    annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    running_max = account + daily_df['cum_net_pnl'].cummax()
    drawdown = (account + daily_df['cum_net_pnl'] - running_max) / running_max
    max_drawdown = drawdown.min()

    # 绘图：累计收益和保证金（分钟级）
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative NetPnL', color=color)
    ax1.plot(plot_df['time'], plot_df['cum_net_pnl'], color=color, label='Cumulative NetPnL')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Margin Occupied', color=color)
    ax2.plot(plot_df['time'], plot_df['margin'], color=color, label='Margin')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.suptitle('Cumulative NetPnL and Margin Over Time')
    fig.tight_layout()
    plt.show()

    # 绘图：每日累计收益
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_df['date'], daily_df['cum_net_pnl'], label='Daily Cumulative NetPnL')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative NetPnL')
    ax.grid(True)
    ax.legend()
    plt.title('Daily Cumulative NetPnL')
    plt.show()

    # 打印指标
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    return daily_df