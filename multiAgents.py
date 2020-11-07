# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


def manhattanDistance(xy1, xy2):
    """
    Function from util.py from previous exercise
    Returns the Manhattan distance between points xy1 and xy2
    """
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

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
        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # return successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        # Idea: Good if you eat food, not good if ghost is too close, good if you go closer to food
        # Closer to closest food?
        foodDistanceListAfter = [manhattanDistance(xy1, newPos) for xy1 in newFood]
        closestFoodDistanceAfter = min(foodDistanceListAfter) if len(foodDistanceListAfter) > 0 else 0

        foodDistanceListBefore = [manhattanDistance(xy1, newPos) for xy1 in oldFood]
        closestFoodDistanceBefore = min(foodDistanceListBefore) if len(foodDistanceListBefore) > 0 else 0

        closerToClosestFood = closestFoodDistanceAfter < closestFoodDistanceBefore
        ateFood = len(newFood) < len(oldFood)

        # Ghost not to close?
        ghostDistanceList = [manhattanDistance(newPos, ghostState.configuration.pos) for ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistanceList)
        ghostToClose = closestGhostDistance < 2

        result = -100 * int(ghostToClose)
        if ateFood:
            return result + 5 * ateFood
        else:
            return result + 3 * int(closerToClosestFood) - closestFoodDistanceAfter


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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value, action = self.value(gameState, 0, self.depth + 1)
        return action

    def value(self, gameState, agent, depth):
        if agent == 0:
            depth -= 1

        # if the state is a terminal state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # return the state's utility
            return self.evaluationFunction(gameState), None
        # if the
        elif agent == 0:  # Pacman is the only max agent
            return self.maxValue(gameState, agent, depth)
        else:
            return self.minValue(gameState, agent, depth)

    def nextAgent(self, gameState, agent):
        return (agent + 1) % gameState.getNumAgents()

    def maxValue(self, gameState, agent, depth):
        # Choices for pacman
        v = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            nextValue = self.value(nextState, self.nextAgent(nextState, agent), depth)[0]
            if nextValue > v:
                v = nextValue
                bestAction = action
        return v, bestAction

    def minValue(self, gameState, agent, depth):
        # Choices for the ghosts
        v = float("inf")
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            v = min(v, self.value(nextState, self.nextAgent(nextState, agent), depth)[0])
        return v, None


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value, action = self.value(gameState, 0, self.depth + 1, -float("inf"), float("inf"))
        return action

    def value(self, gameState, agent, depth, alpha, beta):
        if agent == 0:
            depth -= 1

        # if the state is a terminal state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # return the state's utility
            return self.evaluationFunction(gameState), None
        # if the
        elif agent == 0:  # Pacman is the only max agent
            return self.maxValue(gameState, agent, depth, alpha, beta)
        else:
            return self.minValue(gameState, agent, depth, alpha, beta)

    def nextAgent(self, gameState, agent):
        return (agent + 1) % gameState.getNumAgents()

    def maxValue(self, gameState, agent, depth, alpha, beta):
        # Choices for pacman
        v = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            nextValue = self.value(nextState, self.nextAgent(nextState, agent), depth, alpha, beta)[0]
            if nextValue > v:
                v = nextValue
                bestAction = action
            if v > beta:
                return v, action
            alpha = max(alpha, v)
        return v, bestAction

    def minValue(self, gameState, agent, depth, alpha, beta):
        # Choices for the ghosts
        v = float("inf")
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            v = min(v, self.value(nextState, self.nextAgent(nextState, agent), depth, alpha, beta)[0])
            if v < alpha:
                return v, action
            beta = min(beta, v)
        return v, None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value, action = self.value(gameState, 0, self.depth + 1)
        return action

    def value(self, gameState, agent, depth):
        if agent == 0:
            depth -= 1

        # if the state is a terminal state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # return the state's utility
            return self.evaluationFunction(gameState), None
        # if the
        elif agent == 0:  # Pacman is the only max agent
            return self.maxValue(gameState, agent, depth)
        else:
            return self.minValue(gameState, agent, depth)

    def nextAgent(self, gameState, agent):
        return (agent + 1) % gameState.getNumAgents()

    def maxValue(self, gameState, agent, depth):
        # Choices for pacman
        v = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            nextValue = self.value(nextState, self.nextAgent(nextState, agent), depth)[0]
            if nextValue > v:
                v = nextValue
                bestAction = action
        return v, bestAction

    def minValue(self, gameState, agent, depth):
        # Choices for the ghosts
        v = 0
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            nextValue = self.value(nextState, self.nextAgent(nextState, agent), depth)[0]
            v += float(nextValue) / len(gameState.getLegalActions(agent))
        return v, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Evaluates states, rather than actions like the reflex agent evaluation function did.
    Higher scores = better

    Bad properties get punished (get a negative number)
    Good properties get  rewarded

    The inverse will be taken if good/bad properties are better/worse if they're lower

    Weights define importance/gut feeling/changes after tests

    Bad properties:
    - Food is far (furthestFoodDistance)
    - Not Scared Ghost is close (lower = worse -> inverse)

    Good properties:
    - Not much food left (lower = better -> inverse)
    - Scared ghost is close (lower = better -> inverse)
    - High score
    """
    "*** YOUR CODE HERE ***"
    # return currentGameState.getScore()

    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    notScaredGhosts = [ghostState for ghostState in ghostStates if ghostState.scaredTimer <= 0]
    scaredGhostStates = [ghostState for ghostState in ghostStates if ghostState.scaredTimer > 0]
    capsules = currentGameState.getCapsules()

    foodDistanceList = [manhattanDistance(xy1, pos) for xy1 in food]
    # closestFoodDistance = min(foodDistanceList) if len(foodDistanceList) > 0 else 0
    furthestFoodDistance = max(foodDistanceList) if len(foodDistanceList) > 0 else 9000

    notScaredGhostDistanceList = [manhattanDistance(pos, ghostState.configuration.pos) for ghostState in
                                  notScaredGhosts]
    closestNotScaredGhostDistance = min(notScaredGhostDistanceList) if len(notScaredGhostDistanceList) else -1

    scaredGhostDistanceList = [manhattanDistance(pos, ghostState.configuration.pos) for ghostState in scaredGhostStates]
    closestScaredGhostDistance = min(scaredGhostDistanceList) if len(scaredGhostDistanceList) else -1

    return + (20.0 / len(food) if len(food) else 9000) \
           + 30 / (len(capsules) if len(capsules) else 0.5) \
           - furthestFoodDistance / 20 \
           + (50.0 / closestScaredGhostDistance if closestScaredGhostDistance else 50) \
           - (1.0 / closestNotScaredGhostDistance if closestNotScaredGhostDistance else float("inf")) \
           + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
