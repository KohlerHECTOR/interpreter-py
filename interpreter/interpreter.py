from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.monitor import Monitor
import numpy as np
from copy import deepcopy
from operator import itemgetter

class Interpreter:
    def __init__(self, oracle, tree_policy, env, data_per_iter=5000):
        self.oracle = oracle
        self.tree_policy = tree_policy
        self.data_per_iter = data_per_iter
        self.env = Monitor(env)
        check_for_correct_spaces(env, self.tree_policy.observation_space, self.tree_policy.action_space)
        check_for_correct_spaces(env, self.oracle.observation_space, self.oracle.action_space)


    def train(self, nb_iter):
        print("Fitting tree nb {} ...".format(0))
        S, A = self.oracle.generate_data(self.env, self.data_per_iter)
        self.tree_policy.fit_tree(S, A)
        tree_reward, _ = evaluate_policy(self.tree_policy, self.env)
        
        self.tree_policies = [deepcopy(self.tree_policy)]
        self.tree_policies_rewards = [tree_reward]

        for t in range(nb_iter-1):
            print("Fitting tree nb {} ...".format(t+1))
            S_new, A_new = self.tree_policy.generate_data(self.env, self.data_per_iter)
            S = np.concatenate((S, S_new))
            A = np.concatenate((A, A_new))

            self.tree_policy.fit_tree(S, A)
            tree_reward, _ = evaluate_policy(self.tree_policy, self.env)
            print("Tree policy reward: {}".format(tree_reward))

            self.tree_policies += [deepcopy(self.tree_policy)]
            self.tree_policies_rewards += [tree_reward]


    def get_best_tree_policy(self):
        index, element = max(enumerate(self.tree_policies_rewards), key=itemgetter(1))
        return self.tree_policies[index], element       