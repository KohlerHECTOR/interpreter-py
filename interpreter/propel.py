from .interpreter import Interpreter
from .policies import SB3Policy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3 import DDPG
from torch import no_grad


class CustomTD3PolicyWithProgram(TD3Policy):
    '''
    Implementation of a policy h(s) = f(s) + p(s), where f() is a neural net and p() a program.
    '''
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        program,
        lmb,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.pi = program
        self.lmb = lmb

    def _predict(self, observation, deterministic: bool = False):
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return (
            self.lmb * self.actor(observation)
            + self.pi.predict(observation, deterministic)[0]
        )


class Propel(Interpreter):
    '''
    Implementation of PROPEL.
    Verma et. al. 2019 
    https://proceedings.neurips.cc/paper/2019/hash/5a44a53b7d26bb1e54c05222f186dcfb-Abstract.html
    '''
    def __init__(self, learner, env, **kwargs):
        self.ddpg = DDPG(
            CustomTD3PolicyWithProgram,
            env,
            policy_kwargs=dict(program=learner, lmb=0.1),
            verbose=1,
        )
        super().__init__(SB3Policy(self.ddpg.policy.actor), learner, env, **kwargs)

    def fit_(self, ddpg_steps=1_000, interpreter_steps=1e4):
        with no_grad():
            self.fit(interpreter_steps)
        for _ in range(9):
            self.ddpg.learn(ddpg_steps, progress_bar=True)
            with no_grad():
                self.fit(interpreter_steps)
            self.ddpg.policy.pi = self._policy
