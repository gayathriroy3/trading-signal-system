import numpy as np

def evaluate(trades):
    if len(trades) == 0:
        print("No trades executed")
        return

    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / len(trades)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf

    sharpe = trades.mean() / trades.std() * np.sqrt(len(trades)) if trades.std() != 0 else 0

    final_return = (1 + trades).prod() - 1

    print(f"Trades: {len(trades)}")
    print(f"Final Return: {final_return:.4f}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Win Rate: {win_rate:.2f}")
    print(f"Avg Win: {avg_win:.4f}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Profit Factor: {profit_factor:.2f}")