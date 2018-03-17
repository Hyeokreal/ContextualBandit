# ContextualBandit

## Contextual Bandit Problem

The problem of contextualbandit is implemented in contextual_bandit.py

I defined bandits and actions as a below snippet.

``` self.bandits = np.array([[0., 0.2, -0.1, -0.4], [0.1, -0.5, 0., -0.3], [-0.1, 0., 0., -0.05]]) ```

first index of bandits means context and second index means action(arm)
``` self.bandits[context][action] ```


## Thompson Sampling Approach

If you want to solve Contextual Bandit Problem by Thompson Sampling, It is implemented in thompson_sampling.py

``` $ python thompson_sampling.py ```

## Neural Bandit 1

Also, if you want to solve problem by Neural Network.

[The link of Neural Bandit paper](https://arxiv.org/abs/1409.8191)

``` $ python neural_bandit1.py ```

## TODO

- implementation of NB2
- Modification of SGD in NB1