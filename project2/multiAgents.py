# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

previousAction = 'Stop'


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

    "Add more of your code here if you want to"
    global previousAction
    previousAction = legalMoves[chosenIndex]
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    from util import manhattanDistance
    import copy
    from game import Actions

    global previousAction

    score = 10000
    foodPositions = newFood.asList()
    foodDistance = 0
    tempFoodPositions = copy.deepcopy(foodPositions)
    previousFoodPosition = newPos
    while len(tempFoodPositions) > 0:
      minDistance = 1000000
      for position in tempFoodPositions:
        if manhattanDistance(position, previousFoodPosition) < minDistance:
          minDistance = manhattanDistance(position, previousFoodPosition)
          minDistancePosition = position
      foodDistance += minDistance
      tempFoodPositions.remove(minDistancePosition)
      previousFoodPosition = minDistancePosition

    score -= foodDistance
    for i in xrange(len(newGhostStates)):
      if newScaredTimes[i] <= 0 and manhattanDistance(newGhostStates[i].configuration.getPosition(), newPos) <= 1:
        score -= 1000
    if action == Directions.STOP:
      score -= 100

    if previousAction != 'Stop':
      vector1 = Actions.directionToVector(previousAction)
      vector2 = Actions.directionToVector(action)
      if vector1[0] + vector2[0] == 0 and vector1[1] + vector2[1] == 0:
        score -= 100
    return score


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0  # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def minMaxScore(self, gameState, currentDepth):
    numAgents = gameState.getNumAgents()
    agentIndex = currentDepth % numAgents
    legalMoves = gameState.getLegalActions(agentIndex)
    if len(legalMoves) == 0:
      return self.evaluationFunction(gameState)
    if currentDepth == self.depth * numAgents:
      return self.evaluationFunction(gameState)
    scores = [self.minMaxScore(gameState.generateSuccessor(agentIndex, action), currentDepth + 1) for action in
              legalMoves]
    if agentIndex == 0:
      return max(scores)
    else:
      return min(scores)

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    legalMoves = gameState.getLegalActions(0)
    scores = [self.minMaxScore(gameState.generateSuccessor(0, action), 1) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    print bestScore
    return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def minMaxScore(self, gameState, currentDepth, threshold):
    numAgents = gameState.getNumAgents()
    agentIndex = currentDepth % numAgents
    legalMoves = gameState.getLegalActions(agentIndex)
    if len(legalMoves) == 0:
      return self.evaluationFunction(gameState)
    if currentDepth == self.depth * numAgents:
      return self.evaluationFunction(gameState)
    if agentIndex == 0:
      next_threshold = -1000000
    else:
      next_threshold = 1000000
    scores = []
    for action in legalMoves:
      score = self.minMaxScore(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, next_threshold)
      scores.append(score)
      if agentIndex == 0 and score > threshold:
        break
      if agentIndex == 1 and score < threshold:
        break
      if agentIndex == 0 and score > next_threshold:
        next_threshold = score
      if agentIndex == numAgents - 1 and score < next_threshold:
        next_threshold = score

    if agentIndex == 0:
      return max(scores)
    else:
      return min(scores)

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    legalMoves = gameState.getLegalActions(0)
    next_threshold = -1000000
    scores = []
    for action in legalMoves:
      score = self.minMaxScore(gameState.generateSuccessor(0, action), 1, next_threshold)
      scores.append(score)
      if score >= next_threshold:
        next_threshold = score;
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    print bestScore
    return legalMoves[chosenIndex]


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def minMaxScore(self, gameState, currentDepth):
    numAgents = gameState.getNumAgents()
    agentIndex = currentDepth % numAgents
    legalMoves = gameState.getLegalActions(agentIndex)
    if len(legalMoves) == 0:
      return self.evaluationFunction(gameState)
    if currentDepth + 1 == self.depth * numAgents:
      return self.evaluationFunction(gameState)
    scores = [self.minMaxScore(gameState.generateSuccessor(agentIndex, action), currentDepth + 1) for action in
              legalMoves]
    if agentIndex == 0:
      return max(scores)
    else:
      return sum(scores) / len(legalMoves)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    legalMoves = gameState.getLegalActions(0)
    scores = [self.minMaxScore(gameState.generateSuccessor(0, action), 1) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]


distance_dict = dict()


def calcMazeDistance(point_1, point_2, walls):
  x1, y1 = point_1
  x2, y2 = point_2
  assert not walls[x1][y1], 'point1 is a wall: ' + str(point_1)
  assert not walls[x2][y2], 'point2 is a wall: ' + str(point_2)

  from util import Queue
  from game import Actions
  state_queue = Queue()
  initial_path = []
  state_queue.push((point_1, initial_path))
  visited_states = []
  visited_states.append(point_1)
  while state_queue.isEmpty() == False:
    state = state_queue.pop()
    if state[0] == point_2:
      return len(state[1])
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x, y = state[0]
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = 1
        successors.append((nextState, action, cost))

    for successor in successors:
      if successor[0] in visited_states:
        continue
      visited_states.append(successor[0])
      new_path = state[1][:]
      new_path.append(successor[1])
      state_queue.push((successor[0], new_path))
  return 0


def calcDistance(point_1, point_2, walls):
  global distance_dict
  if (point_1, point_2) in distance_dict:
    return distance_dict[(point_1, point_2)]
  distance = calcMazeDistance(point_1, point_2, walls)
  distance_dict[(point_1, point_2)] = distance
  distance_dict[(point_2, point_1)] = distance
  return distance


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    First, I defined foodDistance as the length of a path linking the pacman and all food. On this path, the next node is always the closest node to the current node. This foodDistance serves as a negative effect. The brave pacman would receive penalty only when it is close enough to a ghost which is not scared. The pacman is also prone to eat nearby capsules.
  """
  "*** YOUR CODE HERE ***"
  import copy

  pacmanPos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  walls = currentGameState.getWalls()

  score = 10000
  foodPositions = food.asList()
  foodDistance = 0
  tempFoodPositions = copy.deepcopy(foodPositions)
  previousFoodPosition = pacmanPos
  while len(tempFoodPositions) > 0:
    minDistance = 1000000
    for position in tempFoodPositions:
      distance = calcDistance(position, previousFoodPosition, walls)
      if distance < minDistance:
        minDistance = distance
        minDistancePosition = position
    foodDistance += minDistance
    tempFoodPositions.remove(minDistancePosition)
    previousFoodPosition = minDistancePosition

  score -= foodDistance
  for i in xrange(len(ghostStates)):
    if scaredTimes[i] <= 0 and manhattanDistance(ghostStates[i].configuration.getPosition(), pacmanPos) <= 1:
      score -= 1000

  score -= len(currentGameState.getCapsules()) * 100

  return score


# Abbreviation
better = betterEvaluationFunction

contestPreviousAction = 'Stop'
contestDistanceDict = dict()
numNeighborsGrid = []
dangerous_points = []
dangerous_ghost_indices = []
previous_num_capsules = 0
previous_num_scared_ghosts = 0


def contestDetectComponents(walls):
  global numNeighborsGrid
  numNeighborsGrid = [[0 for y in range(walls.height)] for x in range(walls.width)]
  for x in xrange(walls.width):
    for y in xrange(walls.height):
      if walls[x][y] == True:
        numNeighborsGrid[x][y] = 0
        continue
      num_neighbors = 0
      for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if walls[x + i][y + j] == False:
          num_neighbors += 1
      numNeighborsGrid[x][y] = num_neighbors


def contestFindPath(point_1, point_2, walls):
  x1, y1 = point_1
  x2, y2 = point_2
  assert not walls[x1][y1], 'point1 is a wall: ' + str(point_1)
  assert not walls[x2][y2], 'point2 is a wall: ' + str(point_2)

  from util import Queue
  from game import Actions
  state_queue = Queue()
  initial_path = []
  initial_path.append(point_1)
  state_queue.push((point_1, initial_path))
  visited_states = []
  visited_states.append(point_1)
  while state_queue.isEmpty() == False:
    state = state_queue.pop()
    if state[0] == point_2:
      return state[1]
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x, y = state[0]
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = 1
        successors.append((nextState, action, cost))

    for successor in successors:
      if successor[0] in visited_states:
        continue
      visited_states.append(successor[0])
      new_path = state[1][:]
      new_path.append(successor[0])
      state_queue.push((successor[0], new_path))
  return []


def contestCalcDistance(point_1, point_2, walls):
  global distance_dict
  if (point_1, point_2) in contestDistanceDict:
    return contestDistanceDict[(point_1, point_2)]
  distance = calcMazeDistance(point_1, point_2, walls)
  contestDistanceDict[(point_1, point_2)] = distance
  contestDistanceDict[(point_2, point_1)] = distance
  return distance


class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

