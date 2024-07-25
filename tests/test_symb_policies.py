from pysr import PySRRegressor

from interpreter import SB3Policy, Interpreter, SymbPolicy
import gymnasium as gym
from stable_baselines3 import PPO
from rlberry.manager import (
    ExperimentManager,
    evaluate_agents,
)
from rlberry.envs import gym_make


def test_symb_policy_ctnuous_actions():
    env = gym.make("Pendulum-v1")
    model = PySRRegressor(binary_operators=["+", "-"], temp_equation_file=True)
    policy = SymbPolicy(model, env)
    s, _ = env.reset()
    policy.predict(s)

def test_symb_policy_discrete_actions():
    env = gym.make("Acrobot-v1")
    model = PySRRegressor(binary_operators=["+", "-"], temp_equation_file=True)
    try:
        policy = SymbPolicy(model, env)
    except AssertionError:
        pass

def test_interpreter_symb_ctnuous_actions_high_dim():
    env = gym.make("Swimmer-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    model = PySRRegressor(binary_operators=["+", "-"], temp_equation_file=True)
    learner = SymbPolicy(model, env)
    interpret = Interpreter(oracle, learner, env)
    interpret.fit(5e3)
    interpret.policy(env.reset()[0])


def test_interpreter_rlberry():
    env = gym.make("Swimmer-v4")
    model = PPO("MlpPolicy", env)
    oracle = SB3Policy(model.policy)
    model = PySRRegressor(binary_operators=["+", "-"], temp_equation_file=True)
    learner = SymbPolicy(model, env)

    exp = ExperimentManager(
        agent_class=Interpreter,
        train_env=(gym_make, {"id": "Swimmer-v4"}),
        fit_budget=1e4,
        init_kwargs=dict(oracle=oracle, learner=learner),
        n_fit=1,
        seed=42,
    )
    exp.fit()

    _ = evaluate_agents(
        [exp], n_simulations=50, show=False
    )  # Evaluate the trained agent on


