import random, util
from game import Agent

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
  return scoreEvaluationFunction(gameState)
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
      numAgents = gameState.getNumAgents()
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

      legalMoves = gameState.getLegalActions()
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

        # BEGIN_YOUR_CODE
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
        # END_YOUR_CODE


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


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



