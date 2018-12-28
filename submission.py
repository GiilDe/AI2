import random, util
from game import Agent, Actions
from util import manhattanDistance
import  numpy as np

def average(arr):
    if len(arr) == 0:
        return 0
    return sum(arr)/len(arr)

def minimum(arr):
    if len(arr) == 0:
        return 0
    return min(arr)

def getGhostsDistances_(gameState):
    pacmanPosition = gameState.getPacmanPosition()
    ghostsPositions = gameState.getGhostPositions()
    distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in ghostsPositions]
    return distancesToPacman

def getGhostsDistances(gameState, isScared):
    pacmanPosition = gameState.getPacmanPosition()
    ghostStates = gameState.getGhostStates()
    if isScared == True:
        ghostsPostitions = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer > 0]
    else:
        ghostsPostitions = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer == 0]
    distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in ghostsPostitions]
    return distancesToPacman


def getCapsulesDistances(gameState):
    pacmanPosition = gameState.getPacmanPosition()
    capsulesPositions = gameState.getCapsules()
    distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in capsulesPositions]
    return distancesToPacman

def getGhostDistAvg(gameState, isScared):
    distancesToPacman = getGhostsDistances(gameState, isScared)
    return average(distancesToPacman)

# returns two distances, with the closest to the left
def getClosestGhostPair(gameState):
    pacmanPosition = gameState.getPacmanPosition()
    distancesToPacman = getGhostsDistances_(gameState)
    if len(distancesToPacman) == 0:
        return None, None
    ghostStates = gameState.getGhostStates()
    closestGhosts = [ghostState for ghostState in ghostStates if
                     manhattanDistance(ghostState.getPosition(), pacmanPosition) == min(distancesToPacman)]
    if len(closestGhosts) > 1:
        return closestGhosts[0], closestGhosts[1]
    else:
        notClosestGhosts = [ghostState for ghostState in ghostStates if
                            manhattanDistance(ghostState.getPosition(), pacmanPosition) != min(distancesToPacman)]
        notClosestDistances = [manhattanDistance(ghostState.getPosition(), pacmanPosition) for
                               ghostState in notClosestGhosts]
        secondClosestGhosts = [ghostState for ghostState in notClosestGhosts if
                               manhattanDistance(ghostState.getPosition(), pacmanPosition) ==
                               min(notClosestDistances)]
        if len(secondClosestGhosts) == 0:
            return closestGhosts[0], None
        return closestGhosts[0], secondClosestGhosts[0]

    # this function is slow as it iterates over the whole board
def getFoodDistances(gameState):
    pacmanPosition = gameState.getPacmanPosition()
    foods = gameState.getFood()
    foodDists = []
    foodsHeight = 0
    for arr in foods:
        foodsHeight += 1
    for i in range(foodsHeight):
        for j in range(len(foods[0])):
            if foods[i][j] is True:
                foodDists.append(manhattanDistance((i, j), pacmanPosition))
    return foodDists

def getAvgFoodDists(gameState):
    foodDists = getFoodDistances(gameState)
    return average(foodDists)

def getMinFoodDist(gameState):
    foodDists = getFoodDistances(gameState)
    return minimum(foodDists)

def getMinCapsuleDist(gameState):
    capsuleDists = getCapsulesDistances(gameState)
    return minimum(capsuleDists)

def getCapsuleDistAvg(gameState):
    capsuleDists = getCapsulesDistances(gameState)
    return average(capsuleDists)


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return scoreEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """
    if gameState.isLose():
        return -float('inf')
    if gameState.isWin():
        return float('inf')
    ghostStates = gameState.getGhostStates()
    isScared = ghostStates[0].scaredTimer > 0
    if isScared is True:
        w = -2
    else:
        w = 2

    closest_ghost, closest_ghost_2 = getClosestGhostPair(gameState)
    avg_2_closest_ghosts = (2 * closest_ghost + closest_ghost_2) / 2
    avg_ghosts_dist = getGhostDistAvg(gameState)
    avg_capsuls_dist = getCapsuleDistAvg(gameState)
    avg_food_dist = getAvgFoodDists(gameState)
    min_capsul_dist = getMinCapsuleDist(gameState)
    min_food_dist = getMinFoodDist(gameState)
    num_food = gameState.getNumFood()
    num_ghosts = gameState.getNumAgents()-1
    score = gameState.getScore()

    #############################################         sqr         ##################################################
    sqr_closest_ghost = closest_ghost**2
    sqr_closest_ghost_2 = closest_ghost_2**2
    sqr_avg_2_closest_ghosts = avg_2_closest_ghosts**2
    sqr_avg_ghosts_dist = avg_ghosts_dist**2
    sqr_avg_capsuls_dist =  avg_capsuls_dist**2
    sqr_avg_food_dist = avg_food_dist**2
    sqr_min_capsul_dist = min_capsul_dist**2
    sqr_min_food_dist = min_food_dist**2
    sqr_num_food = num_food**2
    sqr_num_ghosts = num_ghosts**2
    sqr_score = score**2
    #############################################         sqrt         #################################################
    sqrt_closest_ghost = closest_ghost ** 0.5
    sqrt_closest_ghost_2 = closest_ghost_2 ** 0.5
    sqrt_avg_2_closest_ghosts = avg_2_closest_ghosts ** 0.5
    sqrt_avg_ghosts_dist = avg_ghosts_dist ** 0.5
    sqrt_avg_capsuls_dist = avg_capsuls_dist ** 0.5
    sqrt_avg_food_dist = avg_food_dist ** 0.5
    sqrt_min_capsul_dist = min_capsul_dist ** 0.5
    sqrt_min_food_dist = min_food_dist ** 0.5
    sqrt_num_food = num_food ** 0.5
    sqrt_num_ghosts = num_ghosts ** 0.5
    sqrt_score = score ** 0.5

    ##########################################       food combi         ################################################
    food_comb_1 = - min_food_dist**2/avg_food_dist
    food_comb_2 = -(avg_food_dist**2) * min_food_dist

    ##########################################    good ghost combi      ################################################
    good_ghost_comb_1 = closest_ghost**(-10+avg_2_closest_ghosts)
    good_ghost_comb_2 = avg_2_closest_ghosts**(-10+avg_2_closest_ghosts)
    good_ghost_comb_3 = closest_ghost**(-10+closest_ghost_2)
    good_ghost_comb_4 = closest_ghost_2**(-10+closest_ghost)

    ##########################################    good ghost combi      ################################################
    bad_ghost_comb_1 = - closest_ghost**(3/avg_2_closest_ghosts)
    bad_ghost_comb_2 = - avg_2_closest_ghosts**(3/avg_2_closest_ghosts)

    #############################################    score combi      ##################################################
    score_comb_1 = (score)
    score_comb_2 = (score)**2
    score_comb_3 = (score)**3

    food_val = food_comb_1 + food_comb_2
    score_val = (score_comb_1 + score_comb_2 + score_comb_3)/num_food
    #good_ghost_val = (good_ghost_comb_1 + good_ghost_comb_2 + good_ghost_comb_3 + good_ghost_comb_4)/100
    bad_ghost_val = bad_ghost_comb_1 + bad_ghost_comb_2
    # return food_val + score_val  -bad_ghost_val + min_capsul_dist*avg_food_dist*10

    return score*1000/num_food - closest_ghost*(10 - avg_ghosts_dist) - min_capsul_dist + 1000/avg_food_dist - \
           1000*min_food_dist/num_food + avg_ghosts_dist**2/1000
#     ********* MultiAgent Search Agents- sections c,d,e,f*********
class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

  def getNextAgentIndex(self, index, numAgents):
      return (index + 1) % numAgents

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def minimax(self, gameState, agentIndex, depth):
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

      legalMoves = gameState.getLegalActions(agentIndex)
      if agentIndex == 0:
          curMax = -float('inf')
          for move in legalMoves:
              nextState = gameState.generateSuccessor(agentIndex, move)
              v = self.minimax(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1)
              curMax = max(v, curMax)
          return curMax
      else:
          curMin = float('inf')
          for move in legalMoves:
              nextState = gameState.generateSuccessor(agentIndex, move)
              v = self.minimax(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1)
              curMin = min(v, curMin)
          return curMin


  def getAction(self, gameState):

    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    # BEGIN_YOUR_CODE
    real_depth = self.depth*gameState.getNumAgents()
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = []
    for move in legalMoves:
        nextState = gameState.generateSuccessor(0, move)
        scores.append(self.minimax(nextState, self.getNextAgentIndex(0, nextState.getNumAgents()), real_depth-1))


    bestScore = max(scores)
    bestIndixes = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndixes)  # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_CODE

######################################################################################
# d: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning
      """

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            curMax = -float('inf')
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                v = self.alphaBeta(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1, alpha, beta)
                curMax = max(v, curMax)
                alpha = max(curMax, alpha)
                if curMax >= beta:
                    return float('inf')
            return curMax
        else:
            curMin = float('inf')
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                v = self.alphaBeta(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1, alpha, beta)
                curMin = min(v, curMin)
                beta = min(curMin, beta)
                if curMin <= alpha:
                    return -float('inf')
            return curMin
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        real_depth = self.depth * gameState.getNumAgents()
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = []
        alpha = -float('inf')
        beta = float('inf')
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            v = (self.alphaBeta(nextState, self.getNextAgentIndex(0, nextState.getNumAgents()), real_depth - 1,
                                         alpha, beta))
            scores.append(v)
            alpha = max(v, alpha)

        bestScore = max(scores)
        bestIndixes = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndixes)  # Pick randomly among the best
        return legalMoves[chosenIndex]



######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    def expectimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex != 0: #then its a probabilistic state
            p = 1/len(legalMoves)
            sum = 0
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                sum += p*self.expectimax(nextState, self.getNextAgentIndex(agentIndex,
                                                                         nextState.getNumAgents()), depth - 1)
            return sum
        if agentIndex == 0:
            curMax = -float('inf')
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                v = self.expectimax(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1)
                curMax = max(v, curMax)
            return curMax
        else:
            curMin = float('inf')
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                v = self.expectimax(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1)
                curMin = min(v, curMin)
            return curMin
    """
        Your expectimax agent
      """

    def getAction(self, gameState):
        """
              Returns the expectimax action using self.depth and self.evaluationFunction
              All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """
        real_depth = self.depth * gameState.getNumAgents()
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = []
        alpha = -float('inf')
        beta = float('inf')
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            v = self.expectimax(nextState, self.getNextAgentIndex(0, nextState.getNumAgents()), real_depth - 1)
            scores.append(v)
            alpha = max(v, alpha)

        bestScore = max(scores)
        bestIndixes = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndixes)  # Pick randomly among the best
        return legalMoves[chosenIndex]


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):

    def getDistribution(self, index, state):
        # Read variables from state
        ghostState = state.getGhostState(index)
        legalActions = state.getLegalActions(index)
        pos = state.getGhostPosition(index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        bestProb = 0.8
        if isScared:
            bestScore = max(distancesToPacman)
        else:
            bestScore = min(distancesToPacman)
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


    def expectimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex != 0:  # then its a probabilistic state
            dist = self.getDistribution(agentIndex, gameState)
            sum = 0
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                sum += dist[move] * self.expectimax(nextState, self.getNextAgentIndex(agentIndex,
                                                                             nextState.getNumAgents()), depth - 1)
            return sum
        if agentIndex == 0:
            curMax = -float('inf')
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                v = self.expectimax(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1)
                curMax = max(v, curMax)
            return curMax
        else:
            curMin = float('inf')
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, move)
                v = self.expectimax(nextState, self.getNextAgentIndex(agentIndex, nextState.getNumAgents()), depth - 1)
                curMin = min(v, curMin)
            return curMin

    """
        Your expectimax agent
      """

    def getAction(self, gameState):
        """
              Returns the expectimax action using self.depth and self.evaluationFunction
              All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """
        real_depth = self.depth * gameState.getNumAgents()
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = []
        alpha = -float('inf')
        beta = float('inf')
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            v = self.expectimax(nextState, self.getNextAgentIndex(0, nextState.getNumAgents()), real_depth - 1)
            scores.append(v)
            alpha = max(v, alpha)

        bestScore = max(scores)
        bestIndixes = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndixes)  # Pick randomly among the best
        return legalMoves[chosenIndex]


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def BetterEvaluationFunction(gameState):
      """

      The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

      A GameState specifies the full game state, including the food, capsules, agent configurations and more.
      Following are a few of the helper methods that you can use to query a GameState object to gather information about
      the present state of Pac-Man, the ghosts and the maze:

      gameState.getLegalActions():
      gameState.getPacmanState():
      gameState.getGhostStates():
      gameState.getNumAgents():
      gameState.getScore():
      The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
      """
      if gameState.isLose():
          return -float('inf')
      if gameState.isWin():
          return float('inf')
      ghostStates = gameState.getGhostStates()
      isScared = ghostStates[0].scaredTimer > 0
      if isScared is True:
          w = -2
      else:
          w = 2

      closest_ghost, closest_ghost_2 = getClosestGhostPair(gameState)
      avg_2_closest_ghosts = (2 * closest_ghost + closest_ghost_2) / 2
      avg_ghosts_dist = getGhostDistAvg(gameState)
      avg_capsuls_dist = getCapsuleDistAvg(gameState)
      avg_food_dist = getAvgFoodDists(gameState)
      min_capsul_dist = getMinCapsuleDist(gameState)
      min_food_dist = getMinFoodDist(gameState)
      num_food = gameState.getNumFood()
      num_ghosts = gameState.getNumAgents() - 1
      score = gameState.getScore()

      #############################################         sqr         ##################################################
      sqr_closest_ghost = closest_ghost ** 2
      sqr_closest_ghost_2 = closest_ghost_2 ** 2
      sqr_avg_2_closest_ghosts = avg_2_closest_ghosts ** 2
      sqr_avg_ghosts_dist = avg_ghosts_dist ** 2
      sqr_avg_capsuls_dist = avg_capsuls_dist ** 2
      sqr_avg_food_dist = avg_food_dist ** 2
      sqr_min_capsul_dist = min_capsul_dist ** 2
      sqr_min_food_dist = min_food_dist ** 2
      sqr_num_food = num_food ** 2
      sqr_num_ghosts = num_ghosts ** 2
      sqr_score = score ** 2
      #############################################         sqrt         #################################################
      sqrt_closest_ghost = closest_ghost ** 0.5
      sqrt_closest_ghost_2 = closest_ghost_2 ** 0.5
      sqrt_avg_2_closest_ghosts = avg_2_closest_ghosts ** 0.5
      sqrt_avg_ghosts_dist = avg_ghosts_dist ** 0.5
      sqrt_avg_capsuls_dist = avg_capsuls_dist ** 0.5
      sqrt_avg_food_dist = avg_food_dist ** 0.5
      sqrt_min_capsul_dist = min_capsul_dist ** 0.5
      sqrt_min_food_dist = min_food_dist ** 0.5
      sqrt_num_food = num_food ** 0.5
      sqrt_num_ghosts = num_ghosts ** 0.5
      sqrt_score = score ** 0.5

      ##########################################       food combi         ################################################
      food_comb_1 = min_food_dist ** (6 - num_food)
      food_comb_2 = min_food_dist ** (4 - num_food)
      food_comb_3 = min_food_dist ** (2 - num_food)
      food_comb_4 = min_food_dist ** (1 - num_food)
      food_comb_5 = avg_food_dist ** (10 - num_food)
      food_comb_6 = avg_food_dist ** (8 - num_food)
      food_comb_7 = avg_food_dist ** (5 - num_food)

      #############################################    ghost combi      ##################################################
      ghost_comb_1 = closest_ghost ** (5 - avg_2_closest_ghosts)
      ghost_comb_2 = avg_2_closest_ghosts ** (5 - avg_2_closest_ghosts)
      ghost_comb_3 = closest_ghost ** (5 - closest_ghost_2)
      ghost_comb_4 = closest_ghost_2 ** (5 - closest_ghost)

      #############################################    score combi      ##################################################
      score_comb_1 = (score * w)
      score_comb_2 = (score * w) ** 2
      score_comb_3 = (score * w) ** 3

      ###############################################    with w      #####################################################
      w_closest_ghost = w * closest_ghost
      w_closest_ghost_2 = w * closest_ghost
      w_avg_2_closest_ghosts = w * avg_2_closest_ghosts
      w_avg_ghosts_dist = w * avg_ghosts_dist
      w_avg_capsuls_dist = w * avg_capsuls_dist
      w_avg_food_dist = w * avg_food_dist
      w_min_capsul_dist = w * min_capsul_dist
      w_min_food_dist = w * min_food_dist
      w_num_food = w * num_food
      w_num_ghosts = w * num_ghosts
      w_score = w * score

      w_sqr_closest_ghost = w * sqr_closest_ghost
      w_sqr_closest_ghost_2 = w * sqr_closest_ghost_2
      w_sqr_avg_2_closest_ghosts = w * sqr_avg_2_closest_ghosts
      w_sqr_avg_ghosts_dist = w * sqr_avg_ghosts_dist
      w_sqr_avg_capsuls_dist = w * sqr_avg_capsuls_dist
      w_sqr_avg_food_dist = w * sqr_avg_food_dist
      w_sqr_min_capsul_dist = w * sqr_min_capsul_dist
      w_sqr_min_food_dist = w * sqr_min_food_dist
      w_sqr_num_food = w * sqr_num_food
      w_sqr_num_ghosts = w * sqr_num_ghosts
      w_sqr_score = w * sqr_score

      w_sqrt_closest_ghost = w * sqrt_closest_ghost
      w_sqrt_closest_ghost_2 = w * sqrt_closest_ghost_2
      w_sqrt_avg_2_closest_ghosts = w * sqrt_avg_2_closest_ghosts
      w_sqrt_avg_ghosts_dist = w * sqrt_avg_ghosts_dist
      w_sqrt_avg_capsuls_dist = w * sqrt_avg_capsuls_dist
      w_sqrt_avg_food_dist = w * sqrt_avg_food_dist
      w_sqrt_min_capsul_dist = w * sqrt_min_capsul_dist
      w_sqrt_min_food_dist = w * sqrt_min_food_dist
      w_sqrt_num_food = w * sqrt_num_food
      w_sqrt_num_ghosts = w * sqrt_num_ghosts
      w_sqrt_score = w * sqrt_score

      w_food_comb_1 = w * food_comb_1
      w_food_comb_2 = w * food_comb_2
      w_food_comb_3 = w * food_comb_3
      w_food_comb_4 = w * food_comb_4
      w_food_comb_5 = w * food_comb_5
      w_food_comb_6 = w * food_comb_6
      w_food_comb_7 = w * food_comb_7

      w_ghost_comb_1 = w * ghost_comb_1
      w_ghost_comb_2 = w * ghost_comb_2
      w_ghost_comb_3 = w * ghost_comb_3
      w_ghost_comb_4 = w * ghost_comb_4

      w_score_comb_1 = w * score_comb_1
      w_score_comb_2 = w * score_comb_2
      w_score_comb_3 = w * score_comb_3

      # len = 92
      vals_array = np.array([closest_ghost, closest_ghost_2, avg_ghosts_dist, avg_capsuls_dist, avg_food_dist,
                             min_capsul_dist, min_food_dist, num_food, num_ghosts, score, sqr_closest_ghost,
                             sqr_closest_ghost_2, sqr_avg_2_closest_ghosts, sqr_avg_ghosts_dist, sqr_avg_capsuls_dist,
                             sqr_avg_food_dist, sqr_min_capsul_dist, sqr_min_food_dist, sqr_num_food, sqr_num_ghosts,
                             sqr_score, sqrt_closest_ghost, sqrt_closest_ghost_2, sqrt_avg_2_closest_ghosts,
                             sqrt_avg_ghosts_dist, sqrt_avg_capsuls_dist, sqrt_avg_food_dist, sqrt_min_capsul_dist,
                             sqrt_min_food_dist, sqrt_num_food, sqrt_num_ghosts,
                             sqrt_score, food_comb_1, food_comb_2, food_comb_3, food_comb_4, food_comb_5, food_comb_6,
                             food_comb_7, ghost_comb_1, ghost_comb_2, ghost_comb_3, ghost_comb_4, score_comb_1,
                             score_comb_2, score_comb_3, w_closest_ghost, w_closest_ghost_2, w_avg_ghosts_dist,
                             w_avg_capsuls_dist, w_avg_food_dist, w_min_capsul_dist, w_min_food_dist,
                             w_num_food, w_num_ghosts, w_score, w_sqr_closest_ghost, w_sqr_closest_ghost_2,
                             w_sqr_avg_2_closest_ghosts, w_sqr_avg_ghosts_dist, w_sqr_avg_capsuls_dist,
                             w_sqr_avg_food_dist, w_sqr_min_capsul_dist, w_sqr_min_food_dist, w_sqr_num_food,
                             w_sqr_num_ghosts, w_sqr_score, w_sqrt_closest_ghost, w_sqrt_closest_ghost_2,
                             w_sqrt_avg_2_closest_ghosts, w_sqrt_avg_ghosts_dist, w_sqrt_avg_capsuls_dist,
                             w_sqrt_avg_food_dist, w_sqrt_min_capsul_dist, w_sqrt_min_food_dist, w_sqrt_num_food,
                             w_sqrt_num_ghosts, w_sqrt_score, w_food_comb_1, w_food_comb_2, w_food_comb_3,
                             w_food_comb_4,
                             w_food_comb_5, w_food_comb_6, w_food_comb_7, w_ghost_comb_1, w_ghost_comb_2,
                             w_ghost_comb_3,
                             w_ghost_comb_4, w_score_comb_1, w_score_comb_2, w_score_comb_3])

      from learning import weights
      return vals_array.dot(weights)

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



