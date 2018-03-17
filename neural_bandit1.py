import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.utils.np_utils import to_categorical
from ContextualBandit import ContextualBandit


def build_experts(n, input_shape, n_hidden, n_layers):
    # builds a committee of experts
    def build_expert():
        model = Sequential()
        # add hidden layers
        for layer in range(n_layers):
            model.add(Dense(n_hidden,
                            kernel_initializer='glorot_uniform',
                            activation='relu',
                            input_dim=input_shape,
                            kernel_regularizer=regularizers.l2(0.01)))
        # output layer
        model.add(Dense(1,
                        kernel_initializer='glorot_normal',
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.01)))
        return model

    experts = [build_expert() for i in range(n)]
    return experts


def compile_experts(experts, optimizer, loss):
    # compiles a commitee of experts
    def compile_expert(expert):
        expert.compile(optimizer=optimizer, loss=loss)
        return expert

    compiled_experts = [compile_expert(expert) for expert in experts]
    return compiled_experts


def choose_arm(x, experts, explore):
    n_arms = len(experts)
    # make predictions
    preds = [expert.predict(x) for expert in experts]
    # get best arm
    arm_max = np.nanargmax(preds)
    # create arm selection probabilities
    P = [(1 - explore) * (arm == arm_max) + explore / n_arms for arm in range(n_arms)]
    # select an arm
    chosen_arm = np.random.choice(np.arange(n_arms), p=P)
    pred = preds[chosen_arm]
    return chosen_arm, pred


if __name__ == "__main__":
    '''
    self.bandits = np.array([[0.2, 0., 0., -5], [0.1, -5, 1., 0.25], [-5., 5., 5., 5.]])
    self.bandits = np.array([[0., 0.2, -0.1, -0.4], [0.1, -0.5, 0., -0.3], [-0.1, 0., 0., -0.05]])
    '''
    cb = ContextualBandit()
    num_bandits = cb.num_bandits
    num_actions = cb.num_actions

    model = build_experts(num_actions, num_bandits, 10, 1)
    model[0].summary()

    experts = compile_experts(model, 'adam', 'binary_crossentropy')

    # chosen_arms = []
    # regrets = []
    # true_rewards = []

    count_list = [[0 for _ in range(cb.num_actions)] for _ in range(cb.num_bandits)]

    for itr in range(10000):
        state = cb.get_bandit()
        context = to_categorical(state, num_classes=3)
        action, pred = choose_arm(context, experts, explore=0.001)

        reward = cb.pull_arm(action)

        expert = experts[action]
        expert.fit(context, np.expand_dims(reward, axis=0), epochs=1, verbose=0)

        count_list[state][action] += 1

        if itr % 100:
            print(itr, "th iteration : ", count_list)