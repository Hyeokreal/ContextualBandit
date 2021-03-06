import numpy as np


class ContextualBandit:
    def __init__(self):
        self.state = 0
        # gaussian 에서 뽑은 숫자 보다 작아야 reward가 1이기 때문에 각 bandit에서 숫자가 가장 작은 index의 arm을
        # 당기는 것이 최적입니다.
        self.bandits = np.array([[0., 0.2, -0.1, -0.4], [0.1, -0.5, 0., -0.3], [-0.1, 0., 0., -0.05]])
        # state 0 -> 3 state 1 -> 1 state 2 -> 1 이 최적
        self.num_bandits = np.shape(self.bandits)[0]
        self.num_actions = np.shape(self.bandits)[1]

    def get_bandit(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state

    def pull_arm(self, action):

        bandit_num = self.bandits[self.state][action]
        sampled_number = np.random.randn(1)

        if sampled_number > bandit_num:
            return 1
        else:
            return -1


if __name__ == "__main__":
    cb = ContextualBandit()
    print("The number of bandits", cb.num_bandits)
    print("The number of actions", cb.num_actions)
    print("chosen bandit :", cb.get_bandit())
    total_reward = 0
    for i in range(100):
        cb.state = 2
        reward = cb.pull_arm(0)
        total_reward += reward

    print(total_reward)