# import pylab as P
import numpy as np

trajectories = 1000
trials = 5
bankroll_init = 1000 # dollars
odds = 2.0
P_win = .5
P_lose = 1 - P_win


ending_bankrolls = []
for i in range(trajectories):
    bankroll = bankroll_init
    for j in range(trials):
        # bet_frac = .12
        bet_frac = (odds * P_win  - P_lose)/odds
        bet_size = bet_frac * bankroll
        won = np.random.uniform() > P_win

        if won:
            proceeds = bet_size * odds
        else:
            proceeds = -1 * bet_size

        bankroll += proceeds

        # if we lost our shirt, then stop
        if bankroll <= 0:
            break

        # print("[%4d] won=%-6s bet_frac=%.2f bet_size=%.2f proceeds=%08.2f bankroll=%0.2f" % (i, won, bet_frac, bet_size, proceeds, bankroll))
    print("Ending bankroll %0.2f" % bankroll)
    ending_bankrolls.append(bankroll)

ending_bankrolls = np.array(ending_bankrolls)
outcomes_better_than_start = np.sum(ending_bankrolls > bankroll_init)
outcomes_worst_than_start = np.sum(ending_bankrolls < bankroll_init)
outcomes_lost_it_all = np.sum(ending_bankrolls <= 0.0)
print("Better outcome empirical probability: %s" % (outcomes_better_than_start/float(trajectories)))
print("Worst outcome empirical probability: %s" % (outcomes_worst_than_start/float(trajectories)))
print("Lost it all empirical probability: %s" % (outcomes_lost_it_all/float(trajectories)))
