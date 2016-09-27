def sliding_window(seq, n=3, cycle=True):
    while True:
        it = iter(seq)
        win = deque([it.next() for _ in range(n)], maxlen=n)
        yield win
        for e in it:
            win.append(e)
            yield win
        if not cycle:
            break
