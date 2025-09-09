from __future__ import annotations

import random
import sys as _sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt

# Single import style that works for `python src/week6/rl.py`
_SRC_DIR = Path(__file__).resolve().parents[1]  # points to src/
if str(_SRC_DIR) not in _sys.path:
    _sys.path.append(str(_SRC_DIR))

from week6.heading import *  # noqa: F403
from week6.mdp4e import *  # noqa: F403

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
img = plt.imread(ASSETS_DIR / "grid.png")
plt.imshow(img)

R_s = -0.05

"""
Use R_s to define the maps according to the map given as a nested list, where the external list represents the rows (top down order) and the inner list represents the columns (left to right order). The grey block is marked as None.
Use a pair (a,b) to represent a cell with the first element a representing the column number and the second element b representing the row number. These numbers start from 0. For example, (2,3) represents the cell at the 3rd column and 4th row in the map.
"""
maps: list[list[float | None]] = [
    [R_s, R_s, R_s, R_s, +1],
    [R_s, None, R_s, R_s, -1],
    [R_s, R_s, R_s, R_s, R_s],
    [R_s, R_s, R_s, R_s, R_s],
]

print(maps)

terminals = [(4, 3), (4, 2)]

gamma = 0.95
intended = 0.7
left = 0.2
right = 0.1

sequential_decision_environment = GridMDP(maps, terminals, gamma, intended, left, right)

print(
    f"Display the environment: \n states{sequential_decision_environment.states} \n terminals {sequential_decision_environment.terminals} \n actions {sequential_decision_environment.actlist}\n mdp {sequential_decision_environment.grid}"
)

north = (0, 1)
south = (0, -1)
west = (-1, 0)
east = (1, 0)

print(sequential_decision_environment.states)

# code-swgment 1: Set the intial values of states
U_init: dict[tuple[int, int], float] = dict.fromkeys(sequential_decision_environment.states, 0.0)
U_init[4, 2] = -1.0
U_init[4, 3] = 1.0

print(U_init)

pi = policy_iteration(sequential_decision_environment)
print(pi)

# code-segment 3: Calculate the utility values of states using the policy iteration method
U_values_policy_iteration = policy_evaluation(pi, U_init, sequential_decision_environment, 200)

# code-segment 4: Calculate utility values using the value_iteration algorithm.
U_values_value_iteration = value_iteration(sequential_decision_environment)

# code-segment 5: Display the comparision of estimated utility values from both value-iteration and policy iteration algorithms
temp = sorted(U_values_value_iteration.keys())
print("State, estimated U value using value iteration and policy iteration:\n")
for x in temp:
    print(f"{x},\t{U_values_value_iteration[x]},\t{U_values_policy_iteration[x]}")


class PassiveDUEAgent:
    def __init__(self, pi: Mapping[State, Action | None], mdp: MDP) -> None:
        self.pi: Mapping[State, Action | None] = pi
        self.mdp: MDP = mdp
        self.U: defaultdict[State, float] = defaultdict(float)  # utility estimates
        self.N: defaultdict[State, int] = defaultdict(int)  # total visit counts per state
        self.s: State | None = None
        self.a: Action | None = None
        self.s_history: list[State] = []
        self.r_history: list[float] = []

    def __call__(self, percept: tuple[State, float]) -> Action | None:
        s1, r1 = percept  # reward on entering s1
        self.s_history.append(s1)
        self.r_history.append(r1)

        if s1 in self.mdp.terminals:
            self.s = self.a = None
        else:
            self.s, self.a = s1, self.pi[s1]
        return self.a

    def estimate_U(self) -> dict[State, float]:
        # Call only after an episode ends (we’re in terminal, so self.a is None)
        assert self.a is None, "MDP is not in terminal state"
        assert len(self.s_history) == len(self.r_history)

        gamma: float = getattr(self.mdp, "gamma", 1.0)

        # Compute discounted returns backward
        G: float = 0.0
        # returns_by_state[s] will accumulate all returns observed for s in this episode
        returns_by_state: defaultdict[State, list[float]] = defaultdict(list)

        for t in range(len(self.s_history) - 1, -1, -1):
            G = self.r_history[t] + gamma * G
            s: State = self.s_history[t]
            returns_by_state[s].append(G)

        # Incremental (unbiased) running mean across ALL visits
        for s, G_list in returns_by_state.items():
            for G in G_list:
                self.N[s] += 1
                self.U[s] += (G - self.U[s]) / self.N[s]

        # reset episode buffers
        self.s_history.clear()
        self.r_history.clear()
        return dict(self.U)

    def update_state(self, percept: tuple[State, float]) -> tuple[State, float]:
        return percept


# code-segment 3: Define a class for passive ADP agent
class PassiveADPAgent:
    class ModelMDP(MDP):
        """Class for implementing modified Version of input MDP with
        an editable transition model P and a custom function T.
        """

        def __init__(
            self,
            init: State,
            actlist: Sequence[Action] | Mapping[State, Sequence[Action]],
            terminals: Iterable[State],
            gamma: float,
            states: Iterable[State] | None,
        ) -> None:
            super().__init__(init, actlist, terminals, states=states, gamma=gamma)
            # Transition model learned from experience: (s,a) -> {s': prob}
            self.P: defaultdict[tuple[State, Action | None], defaultdict[State, float]] = (
                defaultdict(lambda: defaultdict(float))
            )
            # Optional: track actions observed per state
            self.A: dict[State, set[Action | None]] = {}

        def T(self, s: State, a: Action | None) -> list[tuple[float, State]]:
            """Return a list of tuples with probabilities for states
            based on the learnt model P.
            """
            return [(prob, res) for (res, prob) in self.P[(s, a)].items()]

    def __init__(self, pi: Mapping[State, Action | None], mdp: MDP) -> None:
        self.pi: Mapping[State, Action | None] = pi
        self.mdp: PassiveADPAgent.ModelMDP = PassiveADPAgent.ModelMDP(
            mdp.init, mdp.actlist, mdp.terminals, mdp.gamma, mdp.states
        )
        self.U: dict[State, float] = {}
        self.Nsa: defaultdict[tuple[State, Action | None], int] = defaultdict(int)
        self.Ns1_sa: defaultdict[tuple[State, State, Action | None], int] = defaultdict(int)
        self.s: State | None = None
        self.a: Action | None = None
        self.visited: set[State] = set()  # keeping track of visited states

    def __call__(self, percept: tuple[State, float]) -> Action | None:
        s1, r1 = percept
        mdp = self.mdp
        R, P, terminals, pi = mdp.reward, mdp.P, mdp.terminals, self.pi
        s, a, Nsa, Ns1_sa, U = self.s, self.a, self.Nsa, self.Ns1_sa, self.U
        # print ("\n s {}, a {}, \n\n Nsa{}, \n\nNs1_sa{}, \n\nU{} ".format( s, a, Nsa, Ns1_sa, U))

        if s1 not in self.visited:  # Reward is only known for visited state.
            U[s1] = 0.0
            R[s1] = r1  # learn the reward for this state
            self.visited.add(s1)

        if s is not None:
            Nsa[(s, a)] += 1
            Ns1_sa[(s1, s, a)] += 1
            # for each t such that Ns′|sa [t, s, a] is nonzero
            for t in [
                res
                for (res, state, act), freq in Ns1_sa.items()
                if (state, act) == (s, a) and freq != 0
            ]:
                P[(s, a)][t] = Ns1_sa[(t, s, a)] / Nsa[(s, a)]
                # print("\nProbability of from {} to \t{} is \t{}".format((s, a), t, P[(s, a)][t]))
            # after observing (s, a)
            if hasattr(mdp, "A"):
                mdp.A.setdefault(s, set()).add(a)
        self.U = policy_evaluation(pi, U, mdp)
        ##
        ##
        self.Nsa, self.Ns1_sa = Nsa, Ns1_sa
        if s1 in terminals:
            self.s = self.a = None
        else:
            self.s, self.a = s1, self.pi[s1]
        # print ("\n s {}, a {}, \n\n Nsa{}, \n\nNs1_sa{}, \n\nU{} ".format( self.s, self.a, Nsa, Ns1_sa, U))
        return self.a

    def update_state(self, percept: tuple[State, float]) -> tuple[State, float]:
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward).
        """
        return percept


class QLearningAgent:
    def __init__(
        self,
        mdp: MDP,
        Ne: int,
        Rplus: float,
        alpha: Callable[[int], float] | None = None,
    ) -> None:
        self.gamma: float = mdp.gamma
        self.terminals: set[State] = set(mdp.terminals)
        self.all_act: Sequence[Action] = cast("Sequence[Action]", mdp.actlist)  # global action list
        self.Ne: int = Ne
        self.Rplus: float = Rplus
        self.Q: defaultdict[tuple[State, Action | None], float] = defaultdict(float)
        # self.Q = defaultdict(lambda: 1/(1.0 - self.gamma))   # optimistic

        self.Nsa: defaultdict[tuple[State, Action | None], int] = defaultdict(int)
        self.s: State | None = None
        self.a: Action | None = None
        # self.alpha = alpha if alpha is not None else (lambda n: 1.0 / (1.0 + n))
        if alpha is not None:
            self.alpha: Callable[[int], float] = alpha
        else:
            alpha0, k = 0.2, 0.001
            self.alpha = lambda n: max(0.02, alpha0 / (1 + k * n))

    def f(self, u: float, n: int) -> float:
        return self.Rplus if n < self.Ne else u

    def actions_in_state(self, state: State) -> list[Action | None]:
        if state in self.terminals:
            return [None]
        return cast("list[Action | None]", list(self.all_act))  # or mdp.actions(state)

    def __call__(self, percept: tuple[State, float]) -> Action | None:
        s1, r1 = self.update_state(percept)

        # Q-learning update for the previous (s,a) using CURRENT reward r1
        if self.s is not None:
            self.Nsa[(self.s, self.a)] += 1
            n = self.Nsa[(self.s, self.a)]
            best_next = max(self.Q[(s1, a1)] for a1 in self.actions_in_state(s1))

            td_target = r1 + self.gamma * best_next
            self.Q[(self.s, self.a)] += self.alpha(n) * (td_target - self.Q[(self.s, self.a)])

        # choose next action (exploration function)
        if s1 in self.terminals:
            self.s = self.a = None
            return None
        self.s = s1
        self.a = max(
            self.actions_in_state(s1), key=lambda a1: self.f(self.Q[(s1, a1)], self.Nsa[(s1, a1)])
        )
        return self.a

    def update_state(self, percept: tuple[State, float]) -> tuple[State, float]:
        return percept


# code-segment 5: Define a function to trial one sequence
def run_single_trial(
    agent_program: Callable[[tuple[State, float]], Action | None] | Any, mdp: MDP
) -> None:
    """Execute trial for given agent_program
    and mdp. mdp should be an instance of subclass
    of mdp.MDP
    """

    def take_single_action(mdp: MDP, s: State, a: Action | None) -> State:
        """Select outcome of taking action a
        in state s. Weighted Sampling.
        """
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for probability_state in mdp.T(s, a):
            probability, state = probability_state
            cumulative_probability += probability
            if x < cumulative_probability:
                break
        return state

    current_state = mdp.init
    sequence: list[tuple[State, float]] = []
    while True:
        current_reward = mdp.R(current_state)
        percept = (current_state, current_reward)
        sequence.append(percept)
        next_action = agent_program(percept)
        if next_action is None:
            # print ("\nSequence{}".format(sequence))
            sequence = []
            break
        current_state = take_single_action(mdp, current_state, next_action)

    if hasattr(agent_program, "estimate_U"):
        results = agent_program.estimate_U()
        # print ("\n utility values {}".format(results))


# code-segment 6: Define a function to convert the Q Values above into U estimates.
def convert_value_estimate(
    states: Mapping[tuple[State, Action | None], float],
) -> dict[State, float]:
    U: defaultdict[State, float] = defaultdict(lambda: -1000.0)  # Large negative for comparison
    for state_action, value in states.items():
        state, _action = state_action
        U[state] = max(U[state], value)
    return dict(U)


num_run = 2000

# Choose a RL agent
model_option = int(
    input(
        "Choose a RL agent from the following: 1-Passive DUEagent, 2-PassiveTDAgent, 3-PassiveADPagent, or 4-Q-LearningAgent \n your choice is: "
    )
)

if model_option == 1:  # Passive DUEagent
    # Create an instance of PassiveDUEAgentclass by calling the constructor PassiveDUEAgent(policy, environment) using the environment created for RL in the environment notebook, env

    DUEagent = PassiveDUEAgent(pi, sequential_decision_environment)

    # Run a number of trials (num_run) for the agent to estimate Utilities using a for-loop
    for i in range(num_run):
        # Display the index of iteration in the for-loop, display every 20 iterations
        if i % 20 == 0:
            print(f"\nTrial {i}\n")

        # Invoke the method run_single_trial(agent-program, decision environment) with the passive DUE agent and environment created from the environment notebook, env
        run_single_trial(DUEagent, sequential_decision_environment)

    # Display the final utility values
    print("\n".join([str(k) + ":" + str(v) for k, v in DUEagent.U.items()]))

    # Display the comparision results
    b = sorted(DUEagent.U.keys())

    print(
        "State, estimated U value using a DUEagent, value iteration and policy iteration are listed below:\n"
    )
    for x in b:
        print(
            f"{x},\t{DUEagent.U[x]},\t{U_values_value_iteration[x]},\t{U_values_policy_iteration[x]}"
        )

elif model_option == 2:  # PassiveTDAgent
    """
    Create an instance of PassiveTDAgentclass by calling the constructor PassiveTDAgent(policy, environment,alpha)
    using the environment created for RL in the environment notebook, env, and
    alpha = lambda n: 60./(59+n)

    """
    TDagent = PassiveTDAgent(pi, sequential_decision_environment, alpha=lambda n: 60.0 / (59 + n))

    # Run a number of trials (`num_run`) for the agent to estimate Utilities using a for-loop
    for i in range(200000):
        # Display the index of iteration in the for-loop, display every 20 iterations
        if i % 2000 == 0:
            print(f"\nTrial {i}\n")
        run_single_trial(TDagent, sequential_decision_environment)

    # Display the final utility values
    print("\n".join([str(k) + ":" + str(v) for k, v in TDagent.U.items()]))

    # Display the comparision results
    b = sorted(U_values_value_iteration.keys())

    print(
        "State, estimated U value using a TDAgent and estimated U value using value iteration are listed below:\n"
    )
    for x in b:
        print(
            f"{x},\t{TDagent.U[x]},\t{U_values_value_iteration[x]},\t{U_values_policy_iteration[x]}"
        )

elif model_option == 3:  # PassiveADPagent
    """
    Create an instance of PassiveADPAgent class by calling the constructor PassiveADPAgent(policy, environment) using the environment created for RL in the environment notebook, env
    """
    ADPagent = PassiveADPAgent(pi, sequential_decision_environment)

    # Run a number of trials (`num_run`) for the agent to estimate Utilities using a for-loop
    for i in range(num_run):
        # Display the index of iteration in the for-loop, display every 20 iterations
        if i % 20 == 0:
            print(f"\nTrial {i}\n")

        # Invoke the method run_single_trial(agent-program, decision environment) with the passive ADP agent and environment created from the environment notebook, env
        run_single_trial(ADPagent, sequential_decision_environment)

    # Display the final utility values
    print("\n".join([str(k) + ":" + str(v) for k, v in ADPagent.U.items()]))

    # Display the comparision results
    b = sorted(U_values_value_iteration.keys())

    print(
        "State, estimated U value using an ADP Agent and estimated U value using value iteration are listed below:\n"
    )
    for x in b:
        print(
            f"{x},\t{ADPagent.U[x]},\t{U_values_value_iteration[x]},\t{U_values_policy_iteration[x]}"
        )

elif model_option == 4:  # Q-LearningAgent
    # Define required parameters to run an exploratory Q-learning agent, Rplus = 2 and Ne = 50
    Rplus = 2
    Ne = 50

    # Use the constructor QLearningAgent(environment, Ne, Rplus, alpha) to create an instance of Q-learning clasth the environment created in the environment notebook, env, parameters used to run an exploratory Q-learning agent, such as Rplus = 2 and Ne = 50 and alpha = lambda n: 60./(59+n) alpha = lambda n: 60./(59+n).

    q_agent = QLearningAgent(
        sequential_decision_environment, Ne, Rplus, alpha=lambda n: 60.0 / (59 + n)
    )

    # Run a number of trials (`num_run`) for the agent to estimate Utilities using a for-loop
    for i in range(num_run):
        # Display the index of iteration in the for-loop, display every 20 iterations
        if i % 40 == 0:
            print(f"\nTrial {i}\n")

        # Invoke the method `run_single_trial(instance of q_agent, decision environment)`, where `decision environment` is the one from the environment notebook, represented by `env`
        run_single_trial(q_agent, sequential_decision_environment)

    # Display the Q values
    print("\n".join([str(c1) + "," + str(c2) for c1, c2 in q_agent.Q.items()]))

    # Convert the Q Values above (q_agent.Q) to Utility values, stored in a variable `U` by calling `convert_value_estimate()` method with parameter: q_agent.Q

    U = convert_value_estimate(q_agent.Q)

    # Display the utility values
    print("\n".join([str(k) + ":" + str(v) for k, v in U.items()]))

    # Display the comparision results
    b = sorted(U_values_value_iteration.keys())

    print(
        "State, estimated U value using a q-Agent and estimated U value using value iteration are listed below:\n"
    )
    for x in b:
        print(
            f"{x},\t{U[x]:9.6f},\t{U_values_value_iteration[x]:9.6f},\t{U_values_policy_iteration[x]:9.6f}"
        )
else:
    print("invalid option number. Try again")
