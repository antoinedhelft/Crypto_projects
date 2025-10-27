import pandas as pd

def backtest_classification(df_test, preds, initial_capital=1000.0, fee=0.001):
    df = df_test.copy()
    df['predicted_signal'] = preds
    capital = initial_capital
    position = 0  # 0 neutre, 1 long, -1 short
    history = []

    for idx, row in df.iterrows():
        sig = row['predicted_signal']
        price = row['close_price']

        # Fermetures / inversions
        if position == 1 and sig == 'Sell':
            capital += price * (1 - fee)
            history.append({'date': idx,'type':'Sell (Close Long)','price':price,'capital':capital})
            position = 0
        elif position == -1 and sig == 'Buy':
            capital -= price * (1 + fee)
            history.append({'date': idx,'type':'Buy (Close Short)','price':price,'capital':capital})
            position = 0

        # Ouvertures
        if position == 0:
            if sig == 'Buy':
                capital -= price * (1 + fee)
                history.append({'date': idx,'type':'Buy (Open Long)','price':price,'capital':capital})
                position = 1
            elif sig == 'Sell':
                capital += price * (1 - fee)
                history.append({'date': idx,'type':'Sell (Open Short)','price':price,'capital':capital})
                position = -1

    # Liquidation finale
    if position != 0:
        final_price = df['close_price'].iloc[-1]
        if position == 1:
            capital += final_price * (1 - fee)
            history.append({'date': df.index[-1],'type':'Final Sell (Long)','price':final_price,'capital':capital})
        else:
            capital -= final_price * (1 + fee)
            history.append({'date': df.index[-1],'type':'Final Buy (Short)','price':final_price,'capital':capital})

    hist_df = pd.DataFrame(history).set_index('date') if history else pd.DataFrame()
    buy_hold = (df['close_price'].iloc[-1]-df['close_price'].iloc[0]) / df['close_price'].iloc[0] * 100
    strat_ret = (capital - initial_capital)/initial_capital * 100
    return {
        'final_capital': capital,
        'strategy_return_pct': strat_ret,
        'buy_hold_return_pct': buy_hold,
        'history': hist_df
    }