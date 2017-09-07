# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"

        import copy
        states = self.mdp.getStates()
        newval = copy.deepcopy(self.values)
        for i in xrange(self.iterations):
            for state in states:
                action = self.getAction(state)
                if action == None:
                    continue
                newval[state] = self.getQValue(state, action)
            self.values = copy.deepcopy(newval)

    def getValue(self, state):
        """
      Return the value of the state (computed in __init__).
    """
        return self.values[state]

    def getQValue(self, state, action):
        """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
        "*** YOUR CODE HERE ***"
        states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for i in xrange(len(states_probs)):
            value = value+ states_probs[i][1] * (
            self.mdp.getReward(state, action, states_probs[i][0]) + self.discount * self.values[states_probs[i][0]])
        return value
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        maxValue = -1000000
        bestAction = None
        for action in actions:
            qval = self.getQValue(state, action)
            if qval > maxValue:
                bestAction = action
                maxValue = qval

        return bestAction
        util.raiseNotDefined()

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)

