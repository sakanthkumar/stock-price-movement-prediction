# Portfolio Risk Analyzer

A comprehensive stock portfolio risk analysis tool that uses Machine Learning to predict stock prices and provide trading signals.

## Features
- **Live Predictions**: Real-time stock price predictions using Gradient Boosting.
- **Technical Analysis**: Advanced indicators (MACD, RSI, Bollinger Bands).
- **Sentiment Analysis**: Incorporates market sentiment (mocked for now).
- **Interactive Dashboard**: React-based frontend for visualization.
- **User Authentication**: Secure login and registration.

## Prerequisites
- Python 3.8+
- Node.js & npm
- MySQL Server (optional for full auth features, but app runs without it)

## Setup & Run Instructions

### 1. Backend Setup

1.  **Navigate to the project root**:
    ```bash
    cd e:/Portfolio-risk-analyzer
    ```

2.  **Create and Activate Virtual Environment** (if not already done):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    - Ensure `.env` file exists in the root directory.
    - It should contain keys for `ALPHAVANTAGE_KEY`, `NEWSAPI_KEY`, etc. (You can use placeholders if you don't have keys yet).

5.  **Run the API**:
    ```bash
    python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.
    - Docs: `http://127.0.0.1:8000/docs`

### 2. Frontend Setup

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```

2.  **Install Dependencies**:
    ```bash
    npm install
    ```

3.  **Start the Frontend**:
    ```bash
    npm start
    ```
    The application will open in your browser at `http://localhost:3000`.

### 3. Usage
- Go to `http://localhost:3000`.
- Login (or register a new account).
- Navigate to the **Dashboard** to see live predictions.
- Click "Run Live Predictions" to fetch the latest data and model outputs.

## Troubleshooting
- **Database Connection**: If you see database errors, ensure your MySQL server is running and the credentials in `.env` are correct. The app will still run in "demo mode" without the database for predictions, but auth will fail.
- **Missing Modules**: If you get `ModuleNotFoundError`, make sure you activated the virtual environment and ran `pip install -r requirements.txt`.
