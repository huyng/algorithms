def gains(principal_amt, buy_price, sell_price):
    """
    Calculates  gains from a given transaction

    :arg principal_amt: Money invested
    :arg buy_price: 
    """

    delta = sell_price - buy_price
    return (delta/float(buy_price)) * principal_amt
