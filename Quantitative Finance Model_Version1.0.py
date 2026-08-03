import pandas as pd
import akshare as ak
import time

# ====================== 全局可调参数（实盘调参仅修改此处） ======================
# SMA均线周期
SMA_FAST = 3
SMA_SLOW = 5
# 简易波动率参数
VOL_WINDOW = 20        # 计算波动率的收益率滚动窗口
VOL_MA_WINDOW = 20     # 判断高低波动的波动率均值窗口
VOL_FILTER_RATIO = 0.6 # 波动衰竭平仓阈值
# 收益率风控阈值
RETURN_LIMIT_UP = 0.07
RETURN_LIMIT_DOWN = -0.07
STOP_LOSS_RATE = -0.06 # 单日6%硬性止损
# 网络容错配置（解决连接断开崩溃核心参数）
REQUEST_TIMEOUT = 18   # 加长超时时间，适配Mac弱网络
MAX_RETRY_TIMES = 3    # 行情拉取失败自动重试3次
RETRY_WAIT_SEC = 1.2   # 每次重试等待1.2秒，降低服务器限流概率

# ====================== 1. API接口获取A股日线数据（新增完整重试+异常捕获，不会直接崩溃） ======================
def generate_demo_kline():
    """
    调用akshare东方财富日线接口获取A股历史后复权数据，修复date列KeyError报错
    内置网络重试机制，连接断开自动重连，多次失败优雅退出不抛崩溃堆栈
    修改stock_code更换股票；start_date/end_date格式YYYYMMDD
    """
    # 股票代码 6位字符串
    stock_code = int(input("The Code of the A-Share"))
    start_date = int(input("The start date of the A-Share"))
    end_date = int(input("The end date of the A-Share"))

    # 循环重试行情接口
    for current_retry in range(MAX_RETRY_TIMES):
        try:
            # 东方财富稳定日线接口，传入超长超时参数
            df_raw = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",       # 日线周期
                start_date=start_date,
                end_date=end_date,
                adjust="hfq",          # hfq=后复权 qfq=前复权
                timeout=REQUEST_TIMEOUT
            )
            # 打印原生列名用于调试
            print("接口原始列名：", df_raw.columns.tolist())

            # 统一字段名：中文列转为全局标准英文列
            df_raw = df_raw.rename(columns={
                "日期": "date",
                "收盘": "close",
                "成交量": "volume"
            })

            # 只保留策略需要的3列，统一顺序
            df = df_raw[["date", "close", "volume"]].copy()

            # 日期升序：旧数据在前，新数据在后（量化计算强制要求）
            df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

            # 转换日期为标准datetime格式
            df["date"] = pd.to_datetime(df["date"])

            # 删除停牌空数据
            df = df.dropna(subset=["close", "volume"])

            # 校验数据长度是否满足指标计算最低窗口
            min_need_len = max(SMA_SLOW, VOL_WINDOW)
            if len(df) < min_need_len:
                print(f"警告：股票{stock_code}历史数据仅{len(df)}根，不足{min_need_len}根，无法完整计算波动率、均线指标")
                return pd.DataFrame()
            # 行情拉取成功，直接返回数据
            return df

        # 捕获所有网络连接、超时、断开类异常
        except Exception as err:
            print(f"【第{current_retry+1}/{MAX_RETRY_TIMES}次行情拉取失败】报错信息：{str(err)}")
            print(f"等待{RETRY_WAIT_SEC}秒后自动重试...")
            time.sleep(RETRY_WAIT_SEC)

    # 全部重试次数耗尽，彻底无法获取行情
    print(f"股票{stock_code}重试{MAX_RETRY_TIMES}次均连接失败，程序终止，建议切换手机热点后重新运行！")
    return pd.DataFrame()

# ====================== 2. 计算SMA3、SMA5 + 金叉死叉标记 ======================
def calc_sma(df: pd.DataFrame) -> pd.DataFrame:
    # 滑动窗口计算简单均线
    df["sma3"] = df["close"].rolling(window=SMA_FAST).mean()
    df["sma5"] = df["close"].rolling(window=SMA_SLOW).mean()

    # 金叉：今日快线>慢线，昨日快线<=慢线
    df["golden_cross"] = (df["sma3"] > df["sma5"]) & (df["sma3"].shift(1) <= df["sma5"].shift(1))
    # 死叉：今日快线<慢线，昨日快线>=慢线
    df["death_cross"] = (df["sma3"] < df["sma5"]) & (df["sma3"].shift(1) >= df["sma5"].shift(1))
    return df

# ====================== 3. 计算单日收益率 + 风控标记 ======================
def calc_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    # 简单收益率 = 今日收盘价 / 昨日收盘价 - 1
    df["daily_return"] = df["close"] / df["close"].shift(1) - 1
    # 前一日收益率是否在安全区间（无±7%极端涨跌）
    df["prev_return_safe"] = (df["daily_return"].shift(1) >= RETURN_LIMIT_DOWN) & (df["daily_return"].shift(1) <= RETURN_LIMIT_UP)
    # 当日触发硬性止损标记
    df["hit_stop_loss"] = df["daily_return"] <= STOP_LOSS_RATE
    return df

# ====================== 4. 核心：计算普通简易波动率（收益率滚动标准差） ======================
def calc_simple_volatility(df: pd.DataFrame) -> pd.DataFrame:
    # 简易波动率：近20日收益率标准差
    df["vol"] = df["daily_return"].rolling(window=VOL_WINDOW).std()
    # 波动率20日均线，作为高低波动判断基准
    df["vol_ma20"] = df["vol"].rolling(window=VOL_MA_WINDOW).mean()
    # 高波动标记：当前波动率 > 波动率20日均（适合开仓）
    df["high_vol"] = df["vol"] > df["vol_ma20"]
    # 波动衰竭标记：当前波动率远低于均值，行情即将横盘（平仓信号）
    df["vol_fade"] = df["vol"] < (df["vol_ma20"] * VOL_FILTER_RATIO)
    return df

# ====================== 5. 生成标准化交易信号（实盘规则落地） ======================
def generate_trade_signal(df: pd.DataFrame) -> pd.DataFrame:
    df["signal"] = "观望"

    # 多头买入：4个条件同时满足
    buy_condition = (
        df["golden_cross"] &
        df["high_vol"] &
        df["prev_return_safe"] &
        (df["close"] > df["sma5"])
    )
    df.loc[buy_condition, "signal"] = "【开仓买入】多头入场"

    # 平仓卖出：满足任意一条立即离场
    sell_condition = (
        df["death_cross"] |
        df["vol_fade"] |
        df["hit_stop_loss"]
    )
    df.loc[sell_condition, "signal"] = "【平仓卖出】全部离场"
    return df

# ====================== 主程序流水线入口 ======================
def main():
    # 1. 调用API加载真实股票行情数据
    df = generate_demo_kline()
    # 行情为空直接终止程序，不执行后续计算
    if len(df) == 0:
        print("无有效行情数据，程序结束")
        return

    print("=" * 90)
    print(f"API获取股票原始数据（共{len(df)}个交易日）")
    print(df.head(10))  # 打印前10行预览
    print("=" * 90)

    # 2. 逐层计算全部指标
    df = calc_sma(df)
    df = calc_daily_return(df)
    df = calc_simple_volatility(df)
    df = generate_trade_signal(df)

    # 3. 精简输出核心指标与交易信号
    show_cols = [
        "date", "close", "sma3", "sma5",
        "daily_return", "vol", "vol_ma20",
        "golden_cross", "death_cross", "high_vol",
        "signal"
    ]
    result_df = df[show_cols].round(4)
    print("完整指标+交易信号总表（后20行）")
    print(result_df.tail(20))
    print("=" * 90)

    # 4. 单独筛选所有买卖点位，复盘专用
    trade_points = df[df["signal"] != "观望"][["date", "close", "signal"]]
    print("所有触发交易点位汇总：")
    print(trade_points)

if __name__ == "__main__":
    main()