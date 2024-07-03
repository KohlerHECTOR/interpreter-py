import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from stable_baselines3.common.utils import is_vectorized_box_observation
from tqdm import tqdm

class Policy(ABC):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
    
    @abstractmethod
    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        raise NotImplementedError
    
    def generate_data(self, env, nb_data):
        assert nb_data > 0 and env.observation_space.shape == self.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            assert env.action_space.n == self.action_space.n
            A = np.zeros((nb_data, 1))
        elif isinstance(env.action_space, gym.spaces.Box):
            assert env.action_space.shape == self.action_space.shape
            A = np.zeros((nb_data, self.action_space.shape[0]))
 
        S = np.zeros((nb_data, self.observation_space.shape[0]))
        s, _ = env.reset()
        for i in tqdm(range(nb_data)):
            action, _ = self.predict(s)
            S[i] = s
            A[i] = action
            s, _, term, trunc, _ = env.step(action)
            if term or trunc:
                s, _ = env.reset()
        return S, A


class SB3Policy(Policy):
    def __init__(self, base_policy):
        self.base_policy = base_policy
        super().__init__(self.base_policy.observation_space, self.base_policy.action_space)
    
    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        return self.base_policy.predict(obs, state, deterministic, episode_start)
        
        
class DTPolicy(Policy):
    def __init__(self, clf, env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        if isinstance(env.action_space, gym.spaces.Box):
            assert isinstance(clf, DecisionTreeRegressor)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            assert isinstance(clf, DecisionTreeClassifier)
        super().__init__(env.observation_space, env.action_space)
        self.clf = clf
        # Policy init
        self.clf.fit(
            [self.observation_space.sample() for _ in range(1000)],
            [self.action_space.sample() for _ in range(1000)],
        )

    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        if not is_vectorized_box_observation(obs, self.observation_space):
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.clf.predict(obs.reshape(1,-1)).squeeze().astype(int)
            else:
                if self.action_space.shape[0] > 1:
                    action = self.clf.predict(obs.reshape(1,-1)).squeeze()
                else:
                    action = self.clf.predict(obs.reshape(1,-1))
            return action, state
        else:
            if isinstance(self.action_space, gym.spaces.Discrete):
                return self.clf.predict(obs).astype(int), None
            else:
                if self.action_space.shape[0] > 1:
                    return self.clf.predict(obs), None
                else:
                    return self.clf.predict(obs)[:,np.newaxis], None
    
    def fit_tree(self, S, A):
        self.clf.fit(S, A)
    

class ObliqueDTPolicy(Policy):
    def __init__(self, clf, env):
        if isinstance(env.action_space, gym.spaces.Box):
            assert isinstance(clf, DecisionTreeRegressor)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            assert isinstance(clf, DecisionTreeClassifier)
        super().__init__(env.observation_space, env.action_space)
        self.clf = clf
        # Policy init
        init_S = np.array([self.observation_space.sample() for _ in range(1000)]).clip(-2, 2)
        self.clf.fit(
            self.get_oblique_data(init_S),
            [self.action_space.sample() for _ in range(1000)],
        )

    def get_oblique_data(self, S):
        # Generate indices for the lower triangular part of the matrix
        indices = np.tril_indices(self.observation_space.shape[0], k=-1)
        # Tile the rows to create matrices for subtraction
        a_mat = np.tile(S[:, np.newaxis, :], (1, self.observation_space.shape[0], 1))
        b_mat = np.transpose(a_mat, axes=(0, 2, 1))

        # Compute the differences and store them in the appropriate location in the result array
        diffs = a_mat - b_mat
        result = diffs[:,  indices[0], indices[1]]

        # Stack the original rows with the differences
        final = np.hstack((S, result))
        return final

    def predict(self, obs, state=None, deterministic=True, episode_start=0):
        if not is_vectorized_box_observation(obs, self.observation_space):
            s_mat = np.tile(obs,(self.observation_space.shape[0],1))
            diff_s = s_mat - s_mat.T
            obs = np.append(obs, diff_s[np.tril_indices(self.observation_space.shape[0], k=-1)])
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.clf.predict(obs.reshape(1,-1)).squeeze().astype(int)
            else:
                if self.action_space.shape[0] > 1:
                    action = self.clf.predict(obs.reshape(1,-1)).squeeze()
                else:
                    action = self.clf.predict(obs.reshape(1,-1))
            return action, state
        else:
            if isinstance(self.action_space, gym.spaces.Discrete):
                return self.clf.predict(self.get_oblique_data(obs)).astype(int), None
            else:
                if self.action_space.shape[0] > 1:
                    return self.clf.predict(self.get_oblique_data(obs)), None
                else:
                    return self.clf.predict(self.get_oblique_data(obs))[:,np.newaxis], None

    def fit_tree(self, S, A):
        self.clf.fit(self.get_oblique_data(S), A)
