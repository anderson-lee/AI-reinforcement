# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for state in self.mdp.getStates():
            self.values[state] = 0        

        for k in range(0, self.iterations):
            batch_values = self.values.copy()
            
            for state in self.mdp.getStates():
                action_vals = []
                if self.mdp.isTerminal(state):
                    continue

                for action in self.mdp.getPossibleActions(state):
                    action_val = 0

                    for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
                        probability = nextState[1]
                        reward = self.mdp.getReward(state, action, nextState[0])
                        vs_prime = self.values[nextState[0]]
                        action_val += probability * (reward + self.discount * vs_prime)

                    action_vals.append(action_val)

                batch_values[state] = max(action_vals)

            self.values = batch_values.copy()
            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q_val = 0

        self.values[state]

        for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val += nextState[1] * (self.mdp.getReward(state, action, nextState[0]) + self.discount * self.values[nextState[0]])
        
        return q_val


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        best_action = ''
        best_score = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            score = self.computeQValueFromValues(state, action)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        for state in self.mdp.getStates():
            self.values[state] = 0        

        for k in range(0, self.iterations):
            # batch_values = self.values.copy()
            
            q = k % len(self.mdp.getStates())
            state = self.mdp.getStates()[q]
            action_vals = []
            if self.mdp.isTerminal(state):
                continue

            for action in self.mdp.getPossibleActions(state):
                action_val = 0

                for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
                    probability = nextState[1]
                    reward = self.mdp.getReward(state, action, nextState[0])
                    vs_prime = self.values[nextState[0]]
                    action_val += probability * (reward + self.discount * vs_prime)

                action_vals.append(action_val)

            self.values[state] = max(action_vals)

            # self.values = self.values.copy()


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        # self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}

        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            predecessors[state] = set()

        # Finding predecessors of each state.
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState[0]].add(state)

        temp_vals = {}

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            maxQVal = float('-inf')
            for action in self.mdp.getPossibleActions(state):
                qVal = self.computeQValueFromValues(state, action)
                if qVal > maxQVal:
                    maxQVal = qVal
            temp_vals[state] = maxQVal
            diff = abs(self.values[state] - maxQVal)

            queue.update(state, -diff)

        for k in range(0, self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = temp_vals[s]
            for p in predecessors[s]:
                maxQVal = float('-inf')
                for action in self.mdp.getPossibleActions(p):
                    qVal = self.computeQValueFromValues(p, action)
                    if qVal > maxQVal:
                        maxQVal = qVal
                temp_vals[p] = maxQVal
                diff = abs(self.getValue(p) - maxQVal)
                if diff > self.theta:
                    queue.update(p, -diff)


        

