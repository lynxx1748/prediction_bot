"""
Data collection for the trading bot.
Fetches market data from exchanges and blockchain.
"""

import time
import requests
import traceback
import logging
from datetime import datetime

from ..core.constants import DB_FILE
from .database import record_market_data

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Collects and processes data from external sources.
    Handles API calls to exchanges and blockchain data collection.
    """
    
    def __init__(self, config):
        """
        Initialize the DataCollector.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.binance_api = "https://api.binance.com/api/v3"
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.db_path = DB_FILE 
        logger.info(f"🔄 Data collector initialized with database at {DB_FILE}")
        
    def get_market_data(self):
        """
        Collect real-time market data from various sources.
        
        Returns:
            dict: Market data or None on failure
        """
        try:
            data = {
                'timestamp': int(time.time()),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Get BNB price and 24h change
            try:
                bnb_response = requests.get(f"{self.binance_api}/ticker/24hr?symbol=BNBUSDT")
                if bnb_response.status_code == 200:
                    bnb_data = bnb_response.json()
                    
                    # Use safer access with defaults
                    data.update({
                        'bnb_price': float(bnb_data.get('lastPrice', bnb_data.get('price', 0))),
                        'bnb_24h_volume': float(bnb_data.get('volume', 0)),
                        'bnb_24h_change': float(bnb_data.get('priceChangePercent', 0)),
                        'bnb_high': float(bnb_data.get('highPrice', 0)),
                        'bnb_low': float(bnb_data.get('lowPrice', 0))
                    })
                else:
                    logger.error(f"❌ Error fetching BNB data: HTTP {bnb_response.status_code}")
                    # Set default values
                    data.update({
                        'bnb_price': 0,
                        'bnb_24h_volume': 0,
                        'bnb_24h_change': 0,
                        'bnb_high': 0,
                        'bnb_low': 0
                    })
            except Exception as e:
                logger.error(f"❌ Error processing BNB data: {e}")
                # Set default values
                data.update({
                    'bnb_price': 0,
                    'bnb_24h_volume': 0,
                    'bnb_24h_change': 0,
                    'bnb_high': 0,
                    'bnb_low': 0
                })
            
            # Get BTC price and 24h change 
            try:
                btc_response = requests.get(f"{self.binance_api}/ticker/24hr?symbol=BTCUSDT")
                if btc_response.status_code == 200:
                    btc_data = btc_response.json()
                    data.update({
                        'btc_price': float(btc_data.get('lastPrice', btc_data.get('price', 0))),
                        'btc_24h_volume': float(btc_data.get('volume', 0)),
                        'btc_24h_change': float(btc_data.get('priceChangePercent', 0))
                    })
                else:
                    logger.error(f"❌ Error fetching BTC data: HTTP {btc_response.status_code}")
                    data.update({
                        'btc_price': 0,
                        'btc_24h_volume': 0,
                        'btc_24h_change': 0
                    })
            except Exception as e:
                logger.error(f"❌ Error processing BTC data: {e}")
                data.update({
                    'btc_price': 0,
                    'btc_24h_volume': 0,
                    'btc_24h_change': 0
                })
            
            # Get Fear & Greed Index
            try:
                fg_data = requests.get("https://api.alternative.me/fng/").json()
                data['fear_greed_value'] = int(fg_data['data'][0]['value'])
                data['fear_greed_class'] = fg_data['data'][0]['value_classification']
            except Exception as e:
                logger.warning(f"⚠️ Error getting Fear & Greed index: {e}")
                data['fear_greed_value'] = 50
                data['fear_greed_class'] = 'Neutral'
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error collecting market data: {e}")
            traceback.print_exc()
            return {'timestamp': int(time.time())} 
            
    def get_onchain_data(self, web3, contract):
        """
        Collect on-chain data from the blockchain.
        
        Args:
            web3: Web3 instance
            contract: Contract instance
            
        Returns:
            dict: On-chain data or None on failure
        """
        try:
            data = {}
            
            # Get contract stats
            total_rounds = contract.functions.currentEpoch().call()
            
            data.update({
                'total_rounds': int(total_rounds),
            })
            
            # Get latest round data
            latest_round = contract.functions.rounds(total_rounds).call()
            data.update({
                'current_epoch': int(total_rounds),
                'totalAmount': float(web3.from_wei(latest_round[8], 'ether')),
                'bullAmount': float(web3.from_wei(latest_round[6], 'ether')),
                'bearAmount': float(web3.from_wei(latest_round[7], 'ether'))
            })
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error collecting on-chain data: {e}")
            traceback.print_exc()
            return None

    def record_data_point(self, market_data, round_data):
        """
        Record a complete data point in the database.
        
        Args:
            market_data: Dictionary with market data
            round_data: Dictionary with round data
            
        Returns:
            bool: Success status
        """
        try:
            if not market_data or not round_data:
                return False
                
            # Calculate bull/bear ratios
            totalAmount = round_data['bullAmount'] + round_data['bearAmount']
            bullRatio = round_data['bullAmount'] / totalAmount if totalAmount > 0 else 0.5
            bearRatio = round_data['bearAmount'] / totalAmount if totalAmount > 0 else 0.5
            
            # Prepare data for database
            data = {
                'timestamp': int(time.time()),
                'bnb_price': market_data.get('bnb_price', 0.0),
                'bnb_24h_volume': market_data.get('bnb_24h_volume', 0.0),
                'bnb_24h_change': market_data.get('bnb_24h_change', 0.0),
                'bnb_high': market_data.get('bnb_high', 0.0),
                'bnb_low': market_data.get('bnb_low', 0.0),
                'btc_price': market_data.get('btc_price', 0.0),
                'btc_24h_volume': market_data.get('btc_24h_volume', 0.0),
                'btc_24h_change': market_data.get('btc_24h_change', 0.0),
                'fear_greed_value': market_data.get('fear_greed_value', 50),
                'fear_greed_class': market_data.get('fear_greed_class', 'Neutral'),
                'total_rounds': round_data.get('total_rounds', 0),
                'current_epoch': round_data.get('current_epoch', 0),
                'totalAmount': totalAmount,
                'bullAmount': round_data['bullAmount'],
                'bearAmount': round_data['bearAmount'],
                'bullRatio': bullRatio,
                'bearRatio': bearRatio,
            }
            
            # Record to database
            record_market_data(data)
            
            # Log summary
            logger.info(f"\n📊 Market Data Recorded:")
            logger.info(f"   Fear & Greed: {data['fear_greed_value']} ({data['fear_greed_class']})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recording data point: {e}")
            traceback.print_exc()
            return False 

    def get_historical_bnb_prices(self, lookback=30, interval='1h'):
        """
        Get historical BNB prices from Binance API.
        
        Args:
            lookback: Number of periods to look back
            interval: Time interval ('1h', '4h', '1d', etc.)
            
        Returns:
            list: List of historical prices or None on failure
        """
        try:
            # Binance API endpoint for klines (candlestick) data
            endpoint = 'https://api.binance.com/api/v3/klines'
            
            # Calculate start and end time
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (lookback * 3600 * 1000)  # lookback hours ago
            
            # Parameters for the API request
            params = {
                'symbol': 'BNBUSDT',  # BNB/USDT pair
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': lookback
            }
            
            # Make the API request
            response = requests.get(endpoint, params=params)
            data = response.json()
            
            if isinstance(data, list):
                # Extract closing prices (4th element in each kline)
                prices = [float(kline[4]) for kline in data]
                return prices
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching historical BNB prices: {e}")
            traceback.print_exc()
            return None 