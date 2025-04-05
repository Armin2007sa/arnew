import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def generate_detailed_analysis(historical_data, signal, symbol, strategy_type="classic"):
    try:
        recent_data = historical_data.tail(5)
        
        signal_type = signal.get('signal', 'NONE')
        
        last_close = recent_data['close'].iloc[-1]
        prev_close = recent_data['close'].iloc[-2]
        price_change = ((last_close - prev_close) / prev_close) * 100
        
        volume = recent_data['volume'].mean()
        
        rsi_value = recent_data['RSI_14'].iloc[-1] if 'RSI_14' in recent_data else 50
        
        trend = "صعودی" if recent_data['close'].pct_change().mean() > 0 else "نزولی"
        
        if strategy_type == "aeai":
            return generate_aeai_analysis(historical_data, signal, symbol, trend, rsi_value, volume, last_close, price_change)
        elif strategy_type == "modern":
            return generate_modern_price_action_analysis(historical_data, signal, symbol, trend, rsi_value, volume, last_close, price_change)
        else:
            # Default: Classic price action analysis
            if signal_type == 'LONG':
                analysis = f"""
بر اساس تحلیل‌های انجام شده، یک سیگنال خرید با اطمینان {signal.get('confidence', 50)}% برای {symbol} شناسایی شده است. 

روند قیمت اخیر {trend} بوده و قیمت در ۲۴ ساعت گذشته حدود {price_change:.2f}% تغییر کرده است. 

شاخص RSI در حال حاضر {rsi_value:.2f} است که نشان‌دهنده {'اشباع خرید' if rsi_value > 70 else 'اشباع فروش' if rsi_value < 30 else 'وضعیت متعادل'} است.

بر اساس سبک‌های معاملاتی برتر جهانی، حجم معاملات به طور متوسط {volume:.2f} بوده که {'بالا و قابل اطمینان' if volume > historical_data['volume'].mean() else 'پایین‌تر از میانگین'} است.

مقاومت‌های مهم پیش رو: {last_close * 1.01:.8f} و {last_close * 1.03:.8f}
حمایت‌های مهم: {last_close * 0.99:.8f} و {last_close * 0.97:.8f}

استراتژی توصیه شده: ورود در قیمت {signal.get('entry', last_close):.8f} با حد ضرر در {signal.get('stop_loss', last_close * 0.97):.8f} و هدف قیمتی {signal.get('take_profit', last_close * 1.05):.8f}.
"""
            elif signal_type == 'SHORT':
                analysis = f"""
بر اساس تحلیل‌های انجام شده، یک سیگنال فروش با اطمینان {signal.get('confidence', 50)}% برای {symbol} شناسایی شده است. 

روند قیمت اخیر {trend} بوده و قیمت در ۲۴ ساعت گذشته حدود {price_change:.2f}% تغییر کرده است. 

شاخص RSI در حال حاضر {rsi_value:.2f} است که نشان‌دهنده {'اشباع خرید' if rsi_value > 70 else 'اشباع فروش' if rsi_value < 30 else 'وضعیت متعادل'} است.

بر اساس سبک‌های معاملاتی برتر جهانی، حجم معاملات به طور متوسط {volume:.2f} بوده که {'بالا و قابل اطمینان' if volume > historical_data['volume'].mean() else 'پایین‌تر از میانگین'} است.

حمایت‌های مهم پیش رو: {last_close * 0.99:.8f} و {last_close * 0.97:.8f}
مقاومت‌های مهم: {last_close * 1.01:.8f} و {last_close * 1.03:.8f}

استراتژی توصیه شده: ورود در قیمت {signal.get('entry', last_close):.8f} با حد ضرر در {signal.get('stop_loss', last_close * 1.03):.8f} و هدف قیمتی {signal.get('take_profit', last_close * 0.95):.8f}.
"""
            else:
                analysis = f"""
در حال حاضر هیچ سیگنال واضحی برای {symbol} شناسایی نشده است.

روند قیمت اخیر {trend} بوده و قیمت در ۲۴ ساعت گذشته حدود {price_change:.2f}% تغییر کرده است.

شاخص RSI در حال حاضر {rsi_value:.2f} است که نشان‌دهنده {'اشباع خرید' if rsi_value > 70 else 'اشباع فروش' if rsi_value < 30 else 'وضعیت متعادل'} است.

توصیه می‌شود منتظر تشکیل الگوهای قیمتی واضح‌تر بمانید یا از استراتژی‌های محافظه‌کارانه استفاده کنید.
"""
            return analysis
            
    except Exception as e:
        logger.error(f"Error generating detailed analysis: {str(e)}")
        return "اطلاعات تحلیلی بیشتر در حال حاضر در دسترس نیست."

def generate_aeai_analysis(historical_data, signal, symbol, trend, rsi_value, volume, last_close, price_change):
    signal_type = signal.get('signal', 'NONE')
    
    # Calculate additional metrics for AEai style
    sma_20 = historical_data['close'].rolling(window=20).mean().iloc[-1] if len(historical_data) >= 20 else last_close
    sma_50 = historical_data['close'].rolling(window=50).mean().iloc[-1] if len(historical_data) >= 50 else last_close
    sma_200 = historical_data['close'].rolling(window=200).mean().iloc[-1] if len(historical_data) >= 200 else last_close
    
    # Detect trend based on moving averages
    ma_trend = "صعودی بلندمدت" if last_close > sma_200 > sma_50 > sma_20 else \
               "صعودی میان‌مدت" if last_close > sma_50 > sma_20 else \
               "صعودی کوتاه‌مدت" if last_close > sma_20 else \
               "نزولی بلندمدت" if last_close < sma_200 < sma_50 < sma_20 else \
               "نزولی میان‌مدت" if last_close < sma_50 < sma_20 else \
               "نزولی کوتاه‌مدت" if last_close < sma_20 else "خنثی"
    
    if signal_type == 'LONG':
        analysis = f"""
🤖 تحلیل هوش مصنوعی AEai (علاءالدین) 🤖

✅ یک سیگنال خرید با اطمینان {signal.get('confidence', 50)}% برای {symbol} شناسایی شده است.

📈 تحلیل روند:
• روند کلی: {ma_trend}
• روند قیمت اخیر: {trend}
• تغییر قیمت ۲۴ ساعته: {price_change:.2f}%

🔎 تحلیل تکنیکال:
• RSI ({rsi_value:.2f}): {'اشباع خرید 🔴' if rsi_value > 70 else 'اشباع فروش 🟢' if rsi_value < 30 else 'متعادل ⚪'}
• حجم معاملات: {'بالا 🟢' if volume > historical_data['volume'].mean() else 'متوسط ⚪' if volume >= historical_data['volume'].mean() * 0.7 else 'پایین 🔴'}
• میانگین متحرک ۲۰: {sma_20:.8f} ({'بالای قیمت 🔴' if sma_20 > last_close else 'زیر قیمت 🟢'})
• میانگین متحرک ۵۰: {sma_50:.8f} ({'بالای قیمت 🔴' if sma_50 > last_close else 'زیر قیمت 🟢'})

💰 استراتژی معاملاتی:
• نقطه ورود: {signal.get('entry', last_close):.8f}
• حد ضرر: {signal.get('stop_loss', last_close * 0.97):.8f} (ریسک: {signal.get('risk_percent', 1)}%)
• هدف قیمتی ۱: {signal.get('take_profit', last_close * 1.03):.8f} (سود: {(signal.get('take_profit', last_close * 1.03) - signal.get('entry', last_close)) / signal.get('entry', last_close) * 100:.2f}%)
• هدف قیمتی ۲: {last_close * 1.05:.8f} (سود: {(last_close * 1.05 - signal.get('entry', last_close)) / signal.get('entry', last_close) * 100:.2f}%)
• هدف قیمتی ۳: {last_close * 1.08:.8f} (سود: {(last_close * 1.08 - signal.get('entry', last_close)) / signal.get('entry', last_close) * 100:.2f}%)
• اهرم پیشنهادی: {signal.get('leverage', 1)}x

⚠️ مدیریت ریسک:
• از اهرم بالاتر از حد پیشنهادی استفاده نکنید
• همیشه از حد ضرر استفاده کنید
• بیش از {signal.get('risk_percent', 1)}% از سرمایه خود را ریسک نکنید

🕰️ تحلیل زمانی:
• بهترین زمان ورود: در صورت تثبیت قیمت بالای {signal.get('entry', last_close):.8f}
• مدت زمان پیشنهادی معامله: ۱-۳ روز

📊 سطوح کلیدی:
• مقاومت اصلی: {last_close * 1.07:.8f}
• مقاومت میانی: {last_close * 1.03:.8f}
• حمایت میانی: {last_close * 0.97:.8f}
• حمایت اصلی: {last_close * 0.93:.8f}
"""
    elif signal_type == 'SHORT':
        analysis = f"""
🤖 تحلیل هوش مصنوعی AEai (علاءالدین) 🤖

🔻 یک سیگنال فروش با اطمینان {signal.get('confidence', 50)}% برای {symbol} شناسایی شده است.

📉 تحلیل روند:
• روند کلی: {ma_trend}
• روند قیمت اخیر: {trend}
• تغییر قیمت ۲۴ ساعته: {price_change:.2f}%

🔎 تحلیل تکنیکال:
• RSI ({rsi_value:.2f}): {'اشباع خرید 🟢' if rsi_value > 70 else 'اشباع فروش 🔴' if rsi_value < 30 else 'متعادل ⚪'}
• حجم معاملات: {'بالا 🟢' if volume > historical_data['volume'].mean() else 'متوسط ⚪' if volume >= historical_data['volume'].mean() * 0.7 else 'پایین 🔴'}
• میانگین متحرک ۲۰: {sma_20:.8f} ({'بالای قیمت 🟢' if sma_20 > last_close else 'زیر قیمت 🔴'})
• میانگین متحرک ۵۰: {sma_50:.8f} ({'بالای قیمت 🟢' if sma_50 > last_close else 'زیر قیمت 🔴'})

💰 استراتژی معاملاتی:
• نقطه ورود: {signal.get('entry', last_close):.8f}
• حد ضرر: {signal.get('stop_loss', last_close * 1.03):.8f} (ریسک: {signal.get('risk_percent', 1)}%)
• هدف قیمتی ۱: {signal.get('take_profit', last_close * 0.97):.8f} (سود: {(signal.get('entry', last_close) - signal.get('take_profit', last_close * 0.97)) / signal.get('entry', last_close) * 100:.2f}%)
• هدف قیمتی ۲: {last_close * 0.95:.8f} (سود: {(signal.get('entry', last_close) - last_close * 0.95) / signal.get('entry', last_close) * 100:.2f}%)
• هدف قیمتی ۳: {last_close * 0.92:.8f} (سود: {(signal.get('entry', last_close) - last_close * 0.92) / signal.get('entry', last_close) * 100:.2f}%)
• اهرم پیشنهادی: {signal.get('leverage', 1)}x

⚠️ مدیریت ریسک:
• از اهرم بالاتر از حد پیشنهادی استفاده نکنید
• همیشه از حد ضرر استفاده کنید
• بیش از {signal.get('risk_percent', 1)}% از سرمایه خود را ریسک نکنید

🕰️ تحلیل زمانی:
• بهترین زمان ورود: در صورت تثبیت قیمت زیر {signal.get('entry', last_close):.8f}
• مدت زمان پیشنهادی معامله: ۱-۳ روز

📊 سطوح کلیدی:
• مقاومت اصلی: {last_close * 1.07:.8f}
• مقاومت میانی: {last_close * 1.03:.8f}
• حمایت میانی: {last_close * 0.97:.8f}
• حمایت اصلی: {last_close * 0.93:.8f}
"""
    else:
        analysis = f"""
🤖 تحلیل هوش مصنوعی AEai (علاءالدین) 🤖

⚠️ در حال حاضر هیچ سیگنال واضحی برای {symbol} شناسایی نشده است.

📊 تحلیل روند:
• روند کلی: {ma_trend}
• روند قیمت اخیر: {trend}
• تغییر قیمت ۲۴ ساعته: {price_change:.2f}%

🔎 تحلیل تکنیکال:
• RSI ({rsi_value:.2f}): {'اشباع خرید 🟡' if rsi_value > 70 else 'اشباع فروش 🟡' if rsi_value < 30 else 'متعادل ⚪'}
• حجم معاملات: {'بالا 🟡' if volume > historical_data['volume'].mean() else 'متوسط ⚪' if volume >= historical_data['volume'].mean() * 0.7 else 'پایین 🟡'}
• میانگین متحرک ۲۰: {sma_20:.8f} ({'بالای قیمت' if sma_20 > last_close else 'زیر قیمت'})
• میانگین متحرک ۵۰: {sma_50:.8f} ({'بالای قیمت' if sma_50 > last_close else 'زیر قیمت'})

⏳ توصیه:
• منتظر تشکیل الگوهای واضح‌تر قیمتی بمانید
• از استراتژی‌های محافظه‌کارانه استفاده کنید
• بازار را به طور مداوم رصد کنید

📊 سطوح کلیدی:
• مقاومت اصلی: {last_close * 1.07:.8f}
• مقاومت میانی: {last_close * 1.03:.8f}
• حمایت میانی: {last_close * 0.97:.8f}
• حمایت اصلی: {last_close * 0.93:.8f}
"""
    
    return analysis

def generate_modern_price_action_analysis(historical_data, signal, symbol, trend, rsi_value, volume, last_close, price_change):
    signal_type = signal.get('signal', 'NONE')
    
    # Calculate additional metrics
    recent_highs = historical_data['high'].rolling(window=5).max().iloc[-5:]
    recent_lows = historical_data['low'].rolling(window=5).min().iloc[-5:]
    
    # Detect higher highs and higher lows (bullish) or lower highs and lower lows (bearish)
    higher_highs = recent_highs.is_monotonic_increasing
    higher_lows = recent_lows.is_monotonic_increasing
    lower_highs = recent_highs.is_monotonic_decreasing
    lower_lows = recent_lows.is_monotonic_decreasing
    
    # Determine price structure
    if higher_highs and higher_lows:
        structure = "روند صعودی قوی (سقف‌های بالاتر و کف‌های بالاتر)"
    elif higher_lows and not higher_highs:
        structure = "روند صعودی ضعیف (کف‌های بالاتر)"
    elif lower_highs and lower_lows:
        structure = "روند نزولی قوی (سقف‌های پایین‌تر و کف‌های پایین‌تر)"
    elif lower_highs and not lower_lows:
        structure = "روند نزولی ضعیف (سقف‌های پایین‌تر)"
    else:
        structure = "رنج (بدون روند مشخص)"
    
    # Check for wicks (shadows) in recent candles
    last_candles = historical_data.iloc[-5:]
    bullish_wicks = ((last_candles['high'] - last_candles['close']) / (last_candles['high'] - last_candles['low'])).mean() < 0.3
    bearish_wicks = ((last_candles['close'] - last_candles['low']) / (last_candles['high'] - last_candles['low'])).mean() < 0.3
    
    if signal_type == 'LONG':
        analysis = f"""
🔍 تحلیل پرایس اکشن مدرن 🔍

✳️ سیگنال: خرید با اطمینان {signal.get('confidence', 50)}% برای {symbol}

📊 ساختار قیمت: {structure}

🕯 تحلیل کندل‌ها:
• سایه‌های کندل: {'مثبت برای خرید (سایه‌های پایینی بلند)' if not bearish_wicks else 'خنثی یا منفی برای خرید'}
• الگوی اخیر: {'تأیید روند صعودی' if higher_lows else 'عدم تأیید قطعی روند'}

📈 سطوح کلیدی:
• سطح ورود: {signal.get('entry', last_close):.8f}
• حد ضرر (ماژور): {signal.get('stop_loss', last_close * 0.97):.8f}
• تاچ پروفیت 1: {signal.get('take_profit', last_close * 1.05):.8f}
• تاچ پروفیت 2: {last_close * 1.08:.8f}

💹 تحلیل حجم:
• حجم معاملات: {'تأییدکننده روند صعودی' if volume > historical_data['volume'].mean() else 'بدون تأیید قوی'}
• نسبت حجم به میانگین: {volume / historical_data['volume'].mean():.2f}

🛡 مدیریت ریسک:
• نسبت ریسک به سود: {(signal.get('take_profit', last_close * 1.05) - signal.get('entry', last_close)) / (signal.get('entry', last_close) - signal.get('stop_loss', last_close * 0.97)):.2f}
• پوزیشن سایز پیشنهادی: {signal.get('risk_percent', 1)}% از سرمایه

⏱ زمان‌بندی:
• ورود: بعد از تثبیت قیمت بالای {signal.get('entry', last_close):.8f}
• خروج: رسیدن به تارگت یا شکست حد ضرر
"""
    elif signal_type == 'SHORT':
        analysis = f"""
🔍 تحلیل پرایس اکشن مدرن 🔍

✳️ سیگنال: فروش با اطمینان {signal.get('confidence', 50)}% برای {symbol}

📊 ساختار قیمت: {structure}

🕯 تحلیل کندل‌ها:
• سایه‌های کندل: {'مثبت برای فروش (سایه‌های بالایی بلند)' if not bullish_wicks else 'خنثی یا منفی برای فروش'}
• الگوی اخیر: {'تأیید روند نزولی' if lower_highs else 'عدم تأیید قطعی روند'}

📉 سطوح کلیدی:
• سطح ورود: {signal.get('entry', last_close):.8f}
• حد ضرر (ماژور): {signal.get('stop_loss', last_close * 1.03):.8f}
• تاچ پروفیت 1: {signal.get('take_profit', last_close * 0.95):.8f}
• تاچ پروفیت 2: {last_close * 0.92:.8f}

💹 تحلیل حجم:
• حجم معاملات: {'تأییدکننده روند نزولی' if volume > historical_data['volume'].mean() else 'بدون تأیید قوی'}
• نسبت حجم به میانگین: {volume / historical_data['volume'].mean():.2f}

🛡 مدیریت ریسک:
• نسبت ریسک به سود: {(signal.get('entry', last_close) - signal.get('take_profit', last_close * 0.95)) / (signal.get('stop_loss', last_close * 1.03) - signal.get('entry', last_close)):.2f}
• پوزیشن سایز پیشنهادی: {signal.get('risk_percent', 1)}% از سرمایه

⏱ زمان‌بندی:
• ورود: بعد از تثبیت قیمت زیر {signal.get('entry', last_close):.8f}
• خروج: رسیدن به تارگت یا شکست حد ضرر
"""
    else:
        analysis = f"""
🔍 تحلیل پرایس اکشن مدرن 🔍

⚠️ سیگنال واضحی برای {symbol} شناسایی نشده است

📊 ساختار قیمت: {structure}

🕯 تحلیل کندل‌ها:
• الگوی سایه‌ها: {'سایه‌های بالایی بلند (نشانه فشار فروش)' if bullish_wicks else 'سایه‌های پایینی بلند (نشانه فشار خرید)' if bearish_wicks else 'بدون الگوی واضح'}
• وضعیت روند: {'انتظار تغییر روند' if price_change > 2 or price_change < -2 else 'تداوم احتمالی روند فعلی'}

📊 سطوح کلیدی برای رصد:
• مقاومت کلیدی: {last_close * 1.03:.8f}
• حمایت کلیدی: {last_close * 0.97:.8f}

💡 توصیه:
• منتظر شکست یکی از سطوح کلیدی بمانید
• به دنبال الگوهای برگشتی یا ادامه‌دهنده باشید
• از معامله در شرایط بدون سیگنال واضح خودداری کنید
"""
    
    return analysis