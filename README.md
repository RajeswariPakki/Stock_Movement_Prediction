# Stock Movement Analysis Based on Social Media Sentiment

This project aims to predict stock market movements based on sentiment analysis of social media posts, specifically tweets and Reddit posts. The project follows a structured approach to analyze sentiment from social media platforms and predict stock prices using machine learning models.

## Step-by-Step Description of the Project

### 1. **Dataset Acquisition**

The project uses two primary datasets for sentiment analysis and stock prediction:
- **`stock_tweets.csv`**: This dataset contains social media tweets about stocks, specifically focusing on sentiment analysis for stock predictions.
  - Columns:
    - `Date`: The date the tweet was posted.
    - `Tweet`: The text content of the tweet.
    - `Company Name`: The name of the company being discussed in the tweet.
    - `Stock Name`: The ticker symbol of the company stock being discussed.
    
- **`stock_yfinance_data.csv`**: This dataset contains historical stock data from Yahoo Finance, which is used to predict stock movements and prices.
  - Columns:
    - `Date`: The date for the stock data.
    - `Open`: The opening price of the stock for the given day.
    - `High`: The highest price of the stock for the given day.
    - `Low`: The lowest price of the stock for the given day.
    - `Close`: The closing price of the stock for the given day.
    - `Adj Close`: The adjusted closing price, accounting for corporate actions like stock splits and dividends.
    - `Volume`: The number of shares traded on that day.
    - `Stock Name`: The ticker symbol of the stock.

### 2. **Sentiment Analysis on Stock Tweets**

The first part of the project involves performing sentiment analysis on the `stock_tweets.csv` dataset to classify tweets based on their sentiment (positive or negative).

#### Steps:
- **Text Preprocessing**: The tweets are preprocessed by removing URLs, special characters, and stopwords, and converting the text to lowercase.
- **Sentiment Analysis with TextBlob**: Using the TextBlob library, the sentiment of each tweet is analyzed. TextBlob provides a `polarity` score, which is a numeric value between -1 (negative) and 1 (positive).
- **Column Addition**: Two new columns are added to the dataset:
  - `sentiment_score`: The numeric polarity score representing the sentiment of each tweet.
  - `sentiment`: A categorical column representing the sentiment, where tweets with positive scores are labeled as "Positive", and those with negative scores are labeled as "Negative".

#### Model Training:
- **Logistic Regression Model**: A Logistic Regression model is trained to classify the sentiment of the tweets. The model achieves an accuracy of 94%, and the results are visualized using a confusion matrix, showing how well the model distinguishes between positive and negative sentiments.

### 3. **Stock Price Prediction Using Random Forest Regressor**

The next part of the project involves predicting stock prices based on historical stock data from `stock_yfinance_data.csv`. The goal is to predict the stock's `Close` price, a continuous numerical value, using a Random Forest Regressor.

#### Steps:
- **Feature Engineering**: A new feature `Prev Close` (previous day's closing price) is created to serve as an input for the model, helping to capture temporal relationships in stock prices.
- **Data Preprocessing**: The dataset is cleaned by converting the `Date` column to datetime format and setting it as the index. Any missing values are dropped (e.g., due to the shift operation).
- **Model Training**: A Random Forest Regressor is trained to predict the `Close` price based on the previous day's closing price (`Prev Close`), as well as other stock features such as `Open`, `High`, `Low`, and `Volume`.
- **Model Evaluation**: The model achieves an excellent R² score of 0.99, indicating highly accurate predictions. The results are visualized to show how well the model predicts stock prices.

### 4. **Stock Movement Prediction Using Random Forest Classifier**

In this part, the project aims to classify whether the stock price will go up or down based on the features from the `stock_yfinance_data.csv` dataset.

#### Steps:
- **Feature Engineering**: A new target column, `Stock Movement`, is created, where:
  - "1" indicates that the stock's `Close` price on the current day is higher than the previous day's `Close`. (Up)
  - "0" indicates that the stock's `Close` price is lower than the previous day's `Close`. (Down)
- **Model Training**: A Random Forest Classifier is used to predict whether the stock's movement will be "Up" or "Down", based on features such as `Open`, `High`, `Low`, `Volume`, and `Prev Close`.
- **Model Evaluation**: The model achieves an accuracy of 75%, and the results are visualized using a confusion matrix, which displays how well the model performs in classifying stock movements.

### 5. **Sentiment Analysis on Reddit Posts**

The final step is to perform sentiment analysis on posts from stock-related subreddits (`r/stocks` and `r/wallstreetbets`) using the trained sentiment analysis model. This analysis helps to understand how social media sentiment may impact stock movements.

#### Steps:
- **Reddit Data Scraping**: Using Reddit's API, relevant posts from the `r/stocks` and `r/wallstreetbets` subreddits are scraped. This data includes the post's title, body text, score (upvotes/downvotes), and creation time.
- **Sentiment Prediction**: The sentiment of each Reddit post is predicted using the pre-trained Logistic Regression model. Each post is classified as either "Positive" or "Negative" based on the sentiment score.
- **Analysis and Insights**: The sentiment data from Reddit is combined with stock data to perform further analysis, including predicting stock movements based on social media sentiment.

### 6. **Conclusion**

This project demonstrates the potential of using machine learning to analyze the impact of social media sentiment on stock market movements. By training models on historical stock data and social media posts, the project offers insights into how sentiment, as expressed in tweets and Reddit posts, may influence stock price movements.

## Datasets Overview

1. **`stock_tweets.csv`**:
   - `Date`: Date of the tweet.
   - `Tweet`: Text content of the tweet.
   - `Company Name`: The company discussed in the tweet.
   - `Stock Name`: The stock ticker symbol for the company.
   - `sentiment_score`: Numeric polarity score (-1 to 1) indicating the sentiment of the tweet.
   - `sentiment`: Categorical sentiment label (Positive, Negative).

2. **`stock_yfinance_data.csv`**:
   - `Date`: Date of the stock data.
   - `Open`: Stock's opening price.
   - `High`: Stock's highest price of the day.
   - `Low`: Stock's lowest price of the day.
   - `Close`: Stock's closing price.
   - `Adj Close`: Adjusted closing price (accounting for splits and dividends).
   - `Volume`: Number of shares traded.
   - `Stock Name`: Stock ticker symbol for the company.

By combining these two datasets, the project builds predictive models that analyze sentiment and stock price movements, providing a comprehensive analysis of how social media sentiment may influence stock performance.
