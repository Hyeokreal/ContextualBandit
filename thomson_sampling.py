import numpy as np
from ContextualBandit import ContextualBandit


class ThompsonBandit:
    def __init__(self, num_bandits, num_actions):
        self.alpha = 0.1
        self.beta = 0.1
        self.num_bandits = num_bandits
        self.num_actions = num_actions

        self.S = [[0 for _ in range(num_actions)] for _ in range(num_bandits)]
        self.F = [[0 for _ in range(num_actions)] for _ in range(num_bandits)]

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
    '''
    self.bandits = np.array([[0.2, 0., 0., -5], [0.1, -5, 1., 0.25], [-5., 5., 5., 5.]])
    '''

    cb = ContextualBandit()
    tb = ThompsonBandit(cb.num_bandits, cb.num_actions)

    count_list = [[0 for _ in range(cb.num_actions)] for _ in range(cb.num_bandits)]
    num_s, num_f = 0, 1

    for iter in range(10000):
        state = cb.get_bandit()
        action = tb.get_action(state)
        reward = cb.pull_arm(action)

        count_list[state][action] += 1

        if reward > 0:
            tb.set_s(state, action)
            num_s += 1
        else:
            tb.set_f(state,action)
            num_f += 1

        if iter % 100 == 0:
            print(iter, "th iteration - success / fail : ", str(num_s/num_f)[:5])


    print(count_list)
    print(tb.S)
    print(tb.F)