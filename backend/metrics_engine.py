import numpy as np
import pandas as pd

class MetricsEngine:
    def __init__(self):
        pass

    def calculate(self, trades: list, equity_curve: list) -> dict:
        """
        Compute professional performance metrics.
        trades: list of dicts { 'pnl': float, 'return_pct': float, 'duration': int }
        equity_curve: list of floats (daily equity values)
        """
        if not trades:
            return {
                "trading_metrics": {
                    "total_trades": 0,
                    "win_rate": "0.00%",
                    "profit_factor": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "net_profit": 0,
                    "expectancy": 0
                },
                "risk_metrics": {
                    "max_drawdown": "0.00%",
                    "sharpe_ratio": 0,
                    "sortino_ratio": 0
                },
                "trades": []
            }

        df_trades = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve)
        
        # Basic Metrics
        total_trades = len(df_trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
        
        # Returns
        net_profit = df_trades['pnl'].sum()
        
        # Drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() # Negative value
        
        # Sharpe Ratio (Daily)
        # Assuming equity curve is daily
        returns = equity_series.pct_change().dropna()
        if returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) # Annualized
        else:
            sharpe = 0
            
        # Sortino Ratio (Downside Risk only)
        downside = returns[returns < 0]
        if downside.std() != 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "trading_metrics": {
                "total_trades": total_trades,
                "win_rate": f"{win_rate*100:.2f}%",
                "profit_factor": round(profit_factor, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "net_profit": round(net_profit, 2),
                "expectancy": round(expectancy, 2)
            },
            "risk_metrics": {
                "max_drawdown": f"{max_drawdown*100:.2f}%",
                "sharpe_ratio": round(sharpe, 2),
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2)
            },
            "trades": trades # Expose raw trades for UI
        }
