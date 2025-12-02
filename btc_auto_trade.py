import os
import time
import schedule
from openai import OpenAI
import openai
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import traceback

def setup_logger():
    """配置日志系统：同时输出到控制台和文件（按日期分割）"""
    log_dir = "trading_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log")

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    class ColorFormatter(logging.Formatter):
        RESET = "\033[0m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        PURPLE = "\033[35m"

        def format(self, record):
            if record.levelno == logging.DEBUG:
                color = self.BLUE
            elif record.levelno == logging.INFO:
                color = self.GREEN
            elif record.levelno == logging.WARNING:
                color = self.YELLOW
            elif record.levelno == logging.ERROR:
                color = self.RED
            elif record.levelno == logging.CRITICAL:
                color = self.PURPLE
            else:
                color = self.RESET

            formatted_msg = super().format(record)
            return f"{color}{formatted_msg}{self.RESET}"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(log_format, date_format))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)

    return logger


logger = setup_logger()

load_dotenv()

deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),
})

TRADE_CONFIG = {
    'symbol': 'BTC/USDT:USDT',
    'amount': 0.008,
    'leverage': 8,
    'timeframe': '15m',
    'test_mode': False,
    'data_points': 120,
    'analysis_periods': {
        'short_term': 18,
        'medium_term': 45,
        'long_term': 100
    }
}

price_history = []
signal_history = []
position = None


def setup_exchange():
    """设置交易所参数"""
    try:
        logger.info("开始初始化交易所参数...")
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}
        )
        logger.info(f"✅ 杠杆倍数设置成功：{TRADE_CONFIG['leverage']}x")

        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        logger.info(f"✅ 账户余额查询成功：USDT {usdt_balance:.2f}")

        return True
    except Exception as e:
        logger.error(f"❌ 交易所设置失败：{str(e)}", exc_info=True)
        return False


def get_current_position():
    """获取当前持仓情况 - OKX版本"""
    try:
        logger.debug("查询当前持仓情况...")
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    position_info = {
                        'side': pos['side'],
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
                    }
                    logger.info(
                        f"✅ 持仓查询成功：{position_info['side']}仓 {position_info['size']} 合约，浮盈 {position_info['unrealized_pnl']:.2f} USDT")
                    return position_info

        logger.info("✅ 持仓查询成功：当前无持仓")
        return None

    except ccxt.BaseError as e:
        logger.error(f"❌ OKX持仓查询接口错误：{str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ 获取持仓失败：{str(e)}", exc_info=True)
    return None


def calculate_technical_indicators(df):
    """计算技术指标"""
    try:
        logger.debug("开始计算技术指标...")

        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)  # 避免除零
        df['rsi'] = 100 - (100 / (1 + rs))

        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 0.001)

        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 0.001)

        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        df = df.bfill().ffill()

        missing_indicators = [col for col in ['sma_5', 'rsi', 'macd', 'bb_position'] if df[col].isna().all()]
        if missing_indicators:
            logger.warning(f"⚠️ 部分技术指标全为NaN：{missing_indicators}")
        else:
            logger.debug("✅ 技术指标计算完成")

        return df
    except Exception as e:
        logger.error(f"❌ 技术指标计算失败：{str(e)}", exc_info=True)
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        logger.debug(f"计算支撑阻力位（回溯周期：{lookback}）...")
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        levels = {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }

        logger.debug(f"✅ 支撑阻力位计算完成：静态阻力={resistance_level:.2f}，静态支撑={support_level:.2f}")
        return levels
    except Exception as e:
        logger.error(f"❌ 支撑阻力计算失败：{str(e)}", exc_info=True)
        return {}


def get_market_trend(df):
    """判断市场趋势"""
    try:
        logger.debug("开始判断市场趋势...")
        current_price = df['close'].iloc[-1]

        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        trend_result = {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }

        logger.debug(f"✅ 趋势判断完成：短期={trend_short}，中期={trend_medium}，整体={overall_trend}")
        return trend_result
    except Exception as e:
        logger.error(f"❌ 趋势分析失败：{str(e)}", exc_info=True)
        return {}


def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        logger.info("开始获取K线数据和技术指标...")
        logger.debug(
            f"调用OKX接口：获取{TRADE_CONFIG['symbol']} {TRADE_CONFIG['timeframe']} K线（数量：{TRADE_CONFIG['data_points']}）")
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                     limit=TRADE_CONFIG['data_points'])

        if not ohlcv:
            logger.error("❌ OKX返回空K线数据")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        logger.debug(f"✅ K线数据获取成功：共{len(df)}根K线")

        df = calculate_technical_indicators(df)

        if len(df) < 2:
            logger.error(f"❌ K线数据不足（仅{len(df)}根），无法计算价格变化")
            return None

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        result = {
            'price': float(current_data['close']), 
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': float(current_data['high']),
            'low': float(current_data['low']),
            'volume': float(current_data['volume']),
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
            'technical_data': {
                'sma_5': float(current_data.get('sma_5', 0)),
                'sma_20': float(current_data.get('sma_20', 0)),
                'sma_50': float(current_data.get('sma_50', 0)),
                'rsi': float(current_data.get('rsi', 0)),
                'macd': float(current_data.get('macd', 0)),
                'macd_signal': float(current_data.get('macd_signal', 0)),
                'macd_histogram': float(current_data.get('macd_histogram', 0)),
                'bb_upper': float(current_data.get('bb_upper', 0)),
                'bb_lower': float(current_data.get('bb_lower', 0)),
                'bb_position': float(current_data.get('bb_position', 0)),
                'volume_ratio': float(current_data.get('volume_ratio', 0))
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }

        logger.info(f"✅ 增强版K线数据处理完成：当前价格=${result['price']:,.2f}，价格变化={result['price_change']:+.2f}%")
        return result
    except ccxt.BaseError as e:
        logger.error(f"❌ OKX交易所接口错误：{str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ 获取增强K线数据失败：{str(e)}", exc_info=True)
    return None


def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    try:
        logger.debug("生成技术分析文本...")
        if 'technical_data' not in price_data:
            logger.warning("⚠️ 技术分析文本生成失败：缺少technical_data字段")
            return "技术指标数据不可用"

        tech = price_data['technical_data']
        trend = price_data.get('trend_analysis', {})
        levels = price_data.get('levels_analysis', {})

        def safe_float(value, default=0):
            return float(value) if value and pd.notna(value) else default

        analysis_text = f"""
        【技术指标分析】
        📈 移动平均线:
        - 5周期: {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
        - 20周期: {safe_float(tech['sma_20']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_20'])) / safe_float(tech['sma_20']) * 100:+.2f}%
        - 50周期: {safe_float(tech['sma_50']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_50'])) / safe_float(tech['sma_50']) * 100:+.2f}%

        🎯 趋势分析:
        - 短期趋势: {trend.get('short_term', 'N/A')}
        - 中期趋势: {trend.get('medium_term', 'N/A')}
        - 整体趋势: {trend.get('overall', 'N/A')}
        - MACD方向: {trend.get('macd', 'N/A')}

        📊 动量指标:
        - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
        - MACD: {safe_float(tech['macd']):.4f}
        - 信号线: {safe_float(tech['macd_signal']):.4f}

        🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

        💰 关键水平:
        - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
        - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}
        """
        logger.debug("✅ 技术分析文本生成完成")
        return analysis_text
    except Exception as e:
        logger.error(f"❌ 技术分析文本生成失败：{str(e)}", exc_info=True)
        return "技术指标数据不可用"


def safe_json_parse(json_str):
    """安全解析JSON"""
    try:
        logger.debug("尝试解析JSON响应...")
        result = json.loads(json_str)
        logger.debug("✅ JSON解析成功")
        return result
    except json.JSONDecodeError:
        try:
            logger.warning(f"⚠️ JSON格式不规范，尝试修复：{json_str[:100]}...")
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            result = json.loads(json_str)
            logger.debug("✅ JSON修复后解析成功")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析失败（修复后仍失败）：{str(e)}", exc_info=True)
            logger.error(f"❌ 原始JSON内容：{json_str}")
            return None


def create_fallback_signal(price_data):
    """创建备用交易信号（修复numpy类型问题）"""
    # 强制转换为普通浮点数，避免numpy类型
    stop_loss = float(price_data['price'] * 0.98)
    take_profit = float(price_data['price'] * 1.02)

    signal = {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": "LOW",
        "is_fallback": True
    }
    logger.warning(f"⚠️ 生成备用交易信号：{signal}")
    return signal


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（彻底修复无持仓格式化错误）"""
    try:
        logger.info("开始使用DeepSeek进行市场分析...")
        technical_analysis = generate_technical_analysis_text(price_data)

        kline_text = f"【最近5根{TRADE_CONFIG['timeframe']}K线数据】\n"
        for i, kline in enumerate(price_data['kline_data'][-5:]):
            trend = "阳线" if kline['close'] > kline['open'] else "阴线"
            change = ((kline['close'] - kline['open']) / kline['open']) * 100
            kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

        signal_text = ""
        if signal_history:
            last_signal = signal_history[-1]
            signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

        current_pos = get_current_position()
        position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

        # 彻底修复：先判断current_pos是否存在，再访问属性（短路求值+统一格式化）
        unrealized_pnl = current_pos['unrealized_pnl'] if current_pos else 0

        prompt = f"""
        你是一个专业的加密货币交易分析师。请基于以下BTC/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

        {kline_text}

        {technical_analysis}

        {signal_text}

        【当前行情】
        - 当前价格: ${price_data['price']:,.2f}
        - 时间: {price_data['timestamp']}
        - 本K线最高: ${price_data['high']:,.2f}
        - 本K线最低: ${price_data['low']:,.2f}
        - 本K线成交量: {price_data['volume']:.2f} BTC
        - 价格变化: {price_data['price_change']:+.2f}%
        - 当前持仓: {position_text}
        - 持仓盈亏: {unrealized_pnl:.2f} USDT  # 安全格式化：已提前处理None情况

        【防频繁交易重要原则】
        1. **趋势持续性优先**: 不要因单根K线或短期波动改变整体趋势判断
        2. **持仓稳定性**: 除非趋势明确强烈反转，否则保持现有持仓方向
        3. **反转确认**: 需要至少2-3个技术指标同时确认趋势反转才改变信号
        4. **成本意识**: 减少不必要的仓位调整，每次交易都有成本

        【交易指导原则 - 必须遵守】
        1. **趋势跟随**: 明确趋势出现时立即行动，不要过度等待
        2. 因为做的是btc，做多权重可以大一点点
        3. **信号明确性**:
        - 强势上涨趋势 → BUY信号
        - 强势下跌趋势 → SELL信号  
        - 仅在窄幅震荡、无明确方向时 → HOLD信号
        4. **技术指标权重**:
        - 趋势(均线排列) > RSI > MACD > 布林带
        - 价格突破关键支撑/阻力位是重要信号

        【当前技术状况分析】
        - 整体趋势: {price_data['trend_analysis'].get('overall', 'N/A')}
        - 短期趋势: {price_data['trend_analysis'].get('short_term', 'N/A')} 
        - RSI状态: {price_data['technical_data'].get('rsi', 0):.1f} ({'超买' if price_data['technical_data'].get('rsi', 0) > 70 else '超卖' if price_data['technical_data'].get('rsi', 0) < 30 else '中性'})
        - MACD方向: {price_data['trend_analysis'].get('macd', 'N/A')}

        【分析要求】
        基于以上分析，请给出明确的交易信号

        请用以下JSON格式回复：
        {{
            "signal": "BUY|SELL|HOLD",
            "reason": "简要分析理由(包含趋势判断和技术依据)",
            "stop_loss": 具体价格,
            "take_profit": 具体价格, 
            "confidence": "HIGH|MEDIUM|LOW"
        }}
        """

        logger.debug(f"发送请求到DeepSeek API，prompt长度：{len(prompt)}字符")
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"您是一位专业的交易员，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        result = response.choices[0].message.content
        logger.debug(f"✅ DeepSeek API响应成功，响应内容：{result[:200]}...")

        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            logger.error(f"❌ 未在DeepSeek响应中找到JSON格式：{result}")
            return create_fallback_signal(price_data)

        json_str = result[start_idx:end_idx]
        signal_data = safe_json_parse(json_str)

        if signal_data is None:
            logger.error("❌ JSON解析失败，触发备用信号")
            return create_fallback_signal(price_data)

        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        missing_fields = [field for field in required_fields if field not in signal_data]
        if missing_fields:
            logger.error(f"❌ DeepSeek返回的JSON缺少必需字段：{missing_fields}，触发备用信号")
            return create_fallback_signal(price_data)

        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        signal_count = len([s for s in signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(signal_history)
        logger.info(
            f"✅ DeepSeek分析完成，生成交易信号：{signal_data['signal']}（最近{total_signals}次中出现{signal_count}次）")

        if len(signal_history) >= 3:
            last_three = [s['signal'] for s in signal_history[-3:]]
            if len(set(last_three)) == 1:
                logger.warning(f"⚠️ 连续3次{signal_data['signal']}信号")

        return signal_data

    except openai.APIError as e:
        logger.error(f"❌ DeepSeek API错误：{str(e)}", exc_info=True)
    except openai.APIConnectionError as e:
        logger.error(f"❌ DeepSeek API连接失败（网络/接口问题）：{str(e)}", exc_info=True)
    except openai.AuthenticationError as e:
        logger.critical(f"❌ DeepSeek API认证失败（API Key无效/过期）：{str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ DeepSeek分析失败：{str(e)}", exc_info=True)
    return create_fallback_signal(price_data)

def execute_trade(signal_data, price_data):
    """执行交易 - OKX版本（修复保证金判断逻辑）"""
    global position

    try:
        logger.info("开始执行交易逻辑...")
        current_position = get_current_position()

        if current_position and signal_data['signal'] != 'HOLD':
            current_side = current_position['side']
            if signal_data['signal'] == 'BUY':
                new_side = 'long'
            elif signal_data['signal'] == 'SELL':
                new_side = 'short'
            else:
                new_side = None

            if new_side != current_side:
                if signal_data['confidence'] != 'HIGH':
                    logger.warning(f"🔒 非高信心反转信号（当前信心：{signal_data['confidence']}），保持现有{current_side}仓")
                    return

                if len(signal_history) >= 2:
                    last_signals = [s['signal'] for s in signal_history[-2:]]
                    if signal_data['signal'] in last_signals:
                        logger.warning(f"🔒 近期已出现{signal_data['signal']}信号，避免频繁反转")
                        return

        logger.info(f"📊 交易信号详情：")
        logger.info(f"  - 信号类型：{signal_data['signal']}")
        logger.info(f"  - 信心程度：{signal_data['confidence']}")
        logger.info(f"  - 分析理由：{signal_data['reason']}")
        logger.info(f"  - 止损价格：${signal_data['stop_loss']:,.2f}")
        logger.info(f"  - 止盈价格：${signal_data['take_profit']:,.2f}")
        logger.info(f"  - 当前持仓：{current_position if current_position else '无'}")

        if signal_data['confidence'] == 'LOW' and not TRADE_CONFIG['test_mode']:
            logger.warning("⚠️ 低信心信号，跳过实盘交易")
            return

        if TRADE_CONFIG['test_mode']:
            logger.info("📌 测试模式 - 仅模拟交易，不实际下单")
            return

        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        # 精准计算所需保证金（加5%缓冲，应对手续费/价格波动）
        required_margin = (price_data['price'] * TRADE_CONFIG['amount'] / TRADE_CONFIG['leverage']) * 1.05

        # 正确判断：可用余额 ≥ 所需保证金（含缓冲）才允许交易
        if usdt_balance < required_margin:
            logger.error(
                f"⚠️ 保证金不足，跳过交易。需要：{required_margin:.2f} USDT（含5%缓冲）, 可用：{usdt_balance:.2f} USDT")
            return
        else:
            logger.info(f"✅ 保证金充足：所需 {required_margin:.2f} USDT（含5%缓冲），可用 {usdt_balance:.2f} USDT")

        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                logger.info("📥 平空仓并开多仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params={'reduceOnly': True}
                )
                time.sleep(1)
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount']
                )
                logger.info("✅ 平空开多完成")
            elif current_position and current_position['side'] == 'long':
                logger.info("📌 已有多头持仓，保持现状")
            else:
                logger.info("📥 开多仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount']
                )
                logger.info("✅ 开多仓完成")

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                logger.info("📤 平多仓并开空仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params={'reduceOnly': True}
                )
                time.sleep(1)
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount']
                )
                logger.info("✅ 平多开空完成")
            elif current_position and current_position['side'] == 'short':
                logger.info("📌 已有空头持仓，保持现状")
            else:
                logger.info("📤 开空仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount']
                )
                logger.info("✅ 开空仓完成")
        else:
            logger.info("📌 HOLD信号，无交易操作")

        time.sleep(2)
        position = get_current_position()
        logger.info(f"✅ 交易执行完成，更新后持仓：{position if position else '无'}")

    except ccxt.InsufficientFunds as e:
        logger.error(f"❌ 交易失败：资金不足 - {str(e)}", exc_info=True)
    except ccxt.OrderNotFound as e:
        logger.error(f"❌ 交易失败：订单未找到 - {str(e)}", exc_info=True)
    except ccxt.BaseError as e:
        logger.error(f"❌ OKX交易接口错误 - {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ 订单执行失败：{str(e)}", exc_info=True)


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek分析第{attempt + 1}/{max_retries}次尝试...")
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            logger.warning(f"第{attempt + 1}次尝试返回备用信号，进行重试...")
            time.sleep(1)

        except Exception as e:
            logger.error(f"第{attempt + 1}次尝试异常：{str(e)}", exc_info=True)
            if attempt == max_retries - 1:
                logger.error(f"✅ 所有重试尝试失败，返回备用信号")
                return create_fallback_signal(price_data)
            time.sleep(1)

    logger.error(f"✅ 重试次数耗尽，返回备用信号")
    return create_fallback_signal(price_data)


def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    next_period_minute = ((current_minute // 15) + 1) * 15
    if next_period_minute == 60:
        next_period_minute = 0

    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        logger.info(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到下一个15分钟整点...")
    else:
        logger.info(f"🕒 等待 {display_seconds} 秒到下一个15分钟整点...")

    return seconds_to_wait


def trading_bot():
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    logger.info("\n" + "=" * 80)
    logger.info(f"📅 开始执行交易周期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    try:
        price_data = get_btc_ohlcv_enhanced()
        if not price_data:
            logger.error("❌ K线数据获取失败，跳过本次交易周期")
            return

        logger.info(
            f"📊 基础行情：BTC当前价格=${price_data['price']:,.2f}，周期={TRADE_CONFIG['timeframe']}，价格变化={price_data['price_change']:+.2f}%")

        signal_data = analyze_with_deepseek_with_retry(price_data)

        if signal_data.get('is_fallback', False):
            logger.warning("⚠️ 当前使用备用交易信号（技术分析流程中断）")

        execute_trade(signal_data, price_data)

        logger.info("=" * 80 + "\n")
    except Exception as e:
        logger.critical(f"❌ 交易周期执行失败：{str(e)}", exc_info=True)
        logger.info("=" * 80 + "\n")


def main():
    logger.info("=" * 80)
    logger.info("🚀 BTC/USDT OKX自动交易机器人启动成功！")
    logger.info("=" * 80)
    logger.info("📋 核心配置：")
    logger.info(f"  - 交易模式：{'模拟模式' if TRADE_CONFIG['test_mode'] else '实盘模式（谨慎操作！）'}")
    logger.info(f"  - 交易标的：{TRADE_CONFIG['symbol']}")
    logger.info(f"  - 交易周期：{TRADE_CONFIG['timeframe']}")
    logger.info(f"  - 杠杆倍数：{TRADE_CONFIG['leverage']}x")
    logger.info(f"  - 交易数量：{TRADE_CONFIG['amount']} BTC/次")
    logger.info(f"  - 执行频率：每15分钟整点执行")
    logger.info("=" * 80)

    if not setup_exchange():
        logger.critical("❌ 交易所初始化失败，程序退出")
        return

    while True:
        trading_bot()
        time.sleep(60)


if __name__ == "__main__":
    main()
