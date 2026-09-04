import os
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI
import uvicorn
import pandas as pd
from datetime import datetime, timedelta
import pytz
from loguru import logger

app = FastAPI(title="ForexFactory News Filter API")

cache = {"data": [], "last_update": None}

def fetch_forex_factory():
    try:
        logger.info("Fetching ForexFactory XML Calendar...")
        # Add headers to bypass simple bot protections
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", headers=headers, timeout=10)
        
        if resp.status_code != 200:
            logger.error(f"Failed to fetch XML. Status: {resp.status_code}")
            return False
            
        root = ET.fromstring(resp.content)
        
        news_list = []
        # FF XML is generally in US Eastern Time (EST/EDT)
        est = pytz.timezone('US/Eastern')
        
        for item in root.findall('event'):
            impact = item.find('impact').text
            if impact != 'High':
                continue
                
            country = item.find('country').text
            date_str = item.find('date').text
            time_str = item.find('time').text
            
            # Skip all-day or tentative events
            if not time_str or time_str.lower() in ['all day', 'tentative']:
                continue
                
            # Parse datetime
            dt_str = f"{date_str} {time_str}"
            try:
                local_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                local_dt = est.localize(local_dt)
                utc_dt = local_dt.astimezone(pytz.utc)
                news_list.append({
                    "currency": country,
                    "time_utc": utc_dt
                })
            except Exception as e:
                pass
                
        cache["data"] = news_list
        cache["last_update"] = datetime.now(pytz.utc)
        logger.info(f"Successfully cached {len(news_list)} High Impact events.")
        return True
    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        return False

@app.get("/news")
def get_news_status(symbol: str = "EURUSD"):
    """
    Evaluates if it is safe to trade based on High Impact news for the given symbol.
    Blocks trading 30 minutes before and 30 minutes after the news.
    """
    now = datetime.now(pytz.utc)
    if cache["last_update"] is None or (now - cache["last_update"]).total_seconds() > 3600:
        fetch_forex_factory()
        
    # Extract currencies (e.g. EURUSD -> EUR, USD. GOLD/XAUUSD -> USD, XAU)
    symbol_upper = symbol.upper().replace("=X", "").replace("-USD", "")
    if symbol_upper == "GOLD" or symbol_upper == "GCF":
        currencies = ["USD"]
    elif len(symbol_upper) >= 6:
        currencies = [symbol_upper[:3], symbol_upper[3:6]]
    else:
        currencies = ["USD"] # Fallback
        
    is_safe = True
    minutes_to_news = 9999.0
    
    for news in cache["data"]:
        if news["currency"] in currencies:
            delta = (news["time_utc"] - now).total_seconds() / 60.0
            
            # Unsafe Zone: -30 mins to +30 mins
            if -30 <= delta <= 30:
                is_safe = False
                minutes_to_news = min(minutes_to_news, delta) if delta > 0 else delta
            elif 0 < delta < minutes_to_news:
                minutes_to_news = delta
                
    return {
        "symbol": symbol,
        "safe": is_safe,
        "minutes_to_next_high_impact": round(minutes_to_news, 1)
    }

if __name__ == "__main__":
    logger.info("Starting News Filter API Gateway on Port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)