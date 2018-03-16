import numpy as np
from ContextualBandit import ContextualBandit


class ThompsonBandit:
    def __init__(self, CB):
        self.alpha = 1
        self.beta = 1
        self.num_bandits = CB.num_bandits
        self.num_actions = CB.num_actions

        self.S = [[0] * self.num_actions] * self.num_bandits  # index 0 -> Success, index 1 - > Fail
        self.F = [[0] * self.num_actions] * self.num_bandits

    def get_action(self, state):
        thetas = [np.random.beta(self.S[state][i] + self.alpha,
                                 self.F[state][i] + self.beta) for i in range(self.num_actions)]
        action = np.argmax(thetas)

        return action

    def set_s(self, state, action):
        self.S[state][action] += 1

    def set_f(self, state, action):
        self.F[state][action] += 1


if __name__ == "__main__":
    cb = ContextualBandit()
    tb = ThompsonBandit(cb)

    num_s, num_f = 0, 1

    for iter in range(10000):
        state = cb.get_bandit()
        action = tb.get_action(state)
        reward = cb.pull_arm(action)

        if reward > 0:
            tb.set_s(state, action)
            num_s += 1
        else:
            tb.set_f(state,action)
            num_f += 1

        if iter % 100 == 0:
            print(iter, "th iteration - success / fail : ", str(num_s/num_f)[:5])