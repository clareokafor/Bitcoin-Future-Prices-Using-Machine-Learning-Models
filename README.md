# Open using Google Colab to view Plotly graphs.

# Bitcoin-Future-Prices-Using-Machine-Learning-Models

This project investigates the prediction of Bitcoin prices using machine learning techniques and historical Bitcoin-GBP data. The study identifies four predictive models; MLP, LSTM, Random Forest, and Gradient Boosting Machine, and selects the top two performers, MLP and Random Forest, for future price predictions. While these models demonstrate promising results, limitations include overfitting, data sensitivity, and overestimation.

Repository: [Bitcoin-Future-Prices-Using-Machine-Learning-Models](https://github.com/clareokafor/Bitcoin-Future-Prices-Using-Machine-Learning-Models)  
Notebook: [BitcoinPricePrediction_pynb.ipynb](https://github.com/clareokafor/Bitcoin-Future-Prices-Using-Machine-Learning-Models/blob/main/Ogochukwu_Okafor_201666459_BitcoinPricePrediction_pynb.ipynb)

---

## 📖 Problem Statement

Stock (and cryptocurrency) price prediction has become central to informed decision-making in modern finance. Bitcoin—particularly in the GBP market, shows extreme volatility driven by supply/demand dynamics, security events, regulatory news and market sentiment. While investors seek to time entries and exits, the unpredictable nature of Bitcoin makes accurate forecasting a formidable time-series challenge. This project tackles that challenge by building, comparing and optimising machine-learning models to forecast future Bitcoin-GBP prices.

## 🎯 Project Objective

- **Understand** historical dynamics of Bitcoin-GBP from 1 January 2015 to 1 September 2023.  
- **Develop** and compare four predictive models—two artificial neural networks (MLP, LSTM) and two ensemble regressors (Random Forest, Gradient Boosting).  
- **Optimise** the top performers (MLP & Random Forest) using hyperparameter search and time-series cross-validation.  
- **Evaluate** models via MSE, RMSE, MAE, MedianAE, R² Score and Explained Variance.  
- **Deliver** robust 30-day ahead price forecasts, and discuss limitations and future improvements.

## 🛠 Approach

1. **Data Acquisition**  
   - Retrieved daily Bitcoin-GBP OHLCV data via Yahoo Finance API (Jan 1 2015 – Sept 1 2023).  

2. **Exploration & Preprocessing**  
   - Visualised trends (line, candlestick, box plots).  
   - Dropped redundant ‘Adj Close’.  
   - Split into features (X) and target (y=Close).  
   - Scaled with Min-Max normalisation; split 80% train / 20% test.

3. **Baseline Modeling**  
   - Trained MLP, LSTM, Random Forest, and Gradient Boosting.  
   - Conducted grid-search CV for hyperparameter tuning.  
   - Measured six metrics to compare baseline performance.

4. **Model Optimization**  
   - Selected MLP & Random Forest as top models from their categories.  
   - Further fine-tuned via time-series cross-validation and grid search.  
   - Retrained and re-evaluated on train/test splits.

5. **Future Forecasting**  
   - Built functions to predict the next 30 days of Bitcoin-GBP prices.  
   - Reincluded ‘Close’ into features for final display of predicted values.

## 🧮 Models & Evaluation

| Model                   | Data Split |    R²    | MSE (×10⁻⁵) | RMSE (×10⁻³) | MAE (×10⁻³) | MedianAE (×10⁻³) | Explained Variance |
|-------------------------|------------|----------|-------------|--------------|-------------|------------------|--------------------|
| **Optimized Random Forest** | Train      | 99.986% | 7.95        | 2.820        | 1.470       | 0.559            | 99.986%            |
|                         | Test       | 99.930% | 3.946       | 6.282        | 2.981       | 0.878            | 99.930%            |
| **Optimized MLP**          | Train      | 99.915% | 4.936       | 7.026        | 3.665       | 1.166            | 99.916%            |
|                         | Test       | 99.929% | 3.946       | 6.282        | 2.981       | 0.878            | 99.929%            |

**Key Insight**: Both optimized models achieve > 99% R² and Explained Variance. Random Forest slightly outperforms MLP in training, while both excel on the hold-out set.

## ⚠️ Limitations

- **Overfitting & Overestimation**: Models tended to overestimate prices at the first future point (Sept 2, 2023).  
- **Data Variability**: Changing end dates alters training size and model behaviour.  
- **Temporal Generalisation**: Random Forest struggles to extrapolate beyond observed covariate ranges; MLP finds long-range temporal patterns challenging.

## 🔭 Future Work

- Integrate **sentiment analysis** (news, social media) as exogenous features.  
- Explore advanced **hyperparameter optimisation** (Bayesian, genetic algorithms).  
- Implement **ensemble of ensembles** and **dynamic feature engineering**.  
- Test **transformer-based** time-series architectures (e.g. Informer, Temporal Fusion Transformer).
