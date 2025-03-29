# 🔮 PredictoBet

<p align="center">
  <img src="https://i.imgur.com/XJyemeI.png" alt="PredictoBet Logo" width="250"/>
  <br>
  <em>Smart Predictions, Smarter Bets</em>
</p>

## Cryptocurrency Prediction Bot for Bull/Bear Markets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-red.svg)]()

A sophisticated prediction and betting system for cryptocurrency binary options markets. PredictoBet uses a combination of machine learning models, technical analysis, and market indicators to predict price movements.

---

## ⚠️ IMPORTANT SECURITY WARNING ⚠️

**A critical bug was discovered that could potentially drain your wallet when running the bot in live mode.** This issue is currently being addressed, but please use EXTREME CAUTION if running this code. It is recommended to:

1. Only run in test mode until the fix is fully deployed and tested
2. Use a dedicated wallet with minimal funds for testing
3. Monitor all transactions carefully

## 🌟 Features

- **Multiple Prediction Strategies**: Combines various prediction techniques to make educated guesses about market direction
- **Adaptive Learning**: Continuously improves through machine learning based on results
- **Self-optimization**: Automatically adjusts strategy weights based on performance
- **Risk Management**: Includes bet sizing algorithms to manage your bankroll
- **Simulation Mode**: Test strategies without risking real money
- **Performance Tracking**: Records all predictions and outcomes for analysis

## 🧠 Prediction Strategies

PredictoBet uses a combination of the following strategies:

1. **Random Forest Model**: Machine learning model trained on historical data
2. **Trend Following**: Analyzes recent price movements to follow established trends
3. **Contrarian Analysis**: Takes positions contrary to market sentiment under specific conditions
4. **Volume Analysis**: Examines trading volume patterns for potential signals
5. **Market Indicators**: Uses standard technical indicators (RSI, MACD, etc.)
6. **Hybrid AI Prediction**: Combines traditional technical analysis with neural network predictions
7. **Technical Analysis**: Applies chart pattern recognition techniques

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Web3 provider configuration
- Trading account on a supported platform
- Crypto wallet with funds for betting (in live mode)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/predictobet.git
   cd predictobet
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration:
   ```
   cp configuration/config.example.json configuration/config.json
   ```
   
4. Edit the configuration file with your settings:
   ```
   nano configuration/config.json
   ```

### Usage

Run the bot in test mode first to ensure everything is working:
Once you're comfortable with how it performs, you can run in live mode (use with caution):

## ⚙️ Configuration

The `config.json` file contains all settings for the bot. Key settings include:

- `trading.betting_enabled`: Set to true to enable actual betting
- `trading.min_confidence`: Minimum confidence threshold for placing bets (0.0-1.0)
- `trading.wager_amount`: Base amount for bets
- `random_forest.parameters`: Machine learning model parameters
- Connection details for the blockchain
- Strategy weights and parameters

## 📊 Performance Analysis

The bot records all predictions and outcomes to a local database. You can analyze performance with:

```
SQLite3
```


## 🔧 Development Status

**ALPHA VERSION**: This project is in active development and may contain bugs or incomplete features. Expect frequent updates and potential breaking changes.

- ✅ Core prediction engine
- ✅ Multiple prediction strategies
- ✅ Database integration
- ✅ Test betting system
- ✅ Machine learning models
- ✅ Error handling
- ✅ Dynamic bet sizing
- ✅ Data flow monitoring 
- 🔄 UI improvements (in progress)
- 🔄 Advanced risk management (in progress)
- 🔄 Additional prediction models (in progress)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Remember**: No prediction system is perfect. Always use caution when trading cryptocurrencies and never risk money you cannot afford to lose.

### Fixed Database Storage Issues

- Improved type handling for blockchain data
- Fixed issues with NULL values in database
- Enhanced data normalization and validation
- Better error handling and diagnostic logging

