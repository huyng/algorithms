def gains(principal_amt, buy_price, sell_price):
    """
    Calculates  gains from a given transaction

    :arg principal_amt: Money invested
    :arg buy_price:
    """

    delta = sell_price - buy_price
    return (delta/float(buy_price)) * principal_amt

def deep_in_the_money_call_return(security_price, strike_price, option_price):
    cost = security_price - option_price
    profit = strike_price - cost
    profit_pct = profit/float(cost)
    return profit_pct


def ditm_call_return(calls):
    calls['Strike'] = calls.index.get_level_values("Strike")
    calls['cost'] = (calls['Underlying_Price'] - calls['Last'])
    calls['profit'] = calls.Strike - calls.cost
    calls['profit_rate'] = calls.profit/calls.cost
    return calls
