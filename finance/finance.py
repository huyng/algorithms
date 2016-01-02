def gains(principal_amt, buy_price, sell_price):
    """
    Calculates  gains from a given transaction

    :arg principal_amt: Money invested
    :arg buy_price:
    """

    delta = sell_price - buy_price
    return (delta/float(buy_price)) * principal_amt

def deep_in_the_money_call_return(security_price, strike_price, call_bid_price):
    cost = security_price - call_bid_price
    profit = strike_price - cost
    profit_pct = profit/float(cost)
    return profit_pct
