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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Initialize all score to 0.
        score = 0

        #Find the distance to the closest food
        disClosestFood = float("inf")
        allFood = newFood.asList()
        for food in allFood:
            tempFoodDis = util.manhattanDistance(newPos, food)
            if tempFoodDis < disClosestFood:
                disClosestFood = tempFoodDis
        score -= disClosestFood # we want the pacman which is closer to the food to reduce the movement

        #Find whether pacman in the new position will meet the ghost or not. If pacman will meet the ghost, it is the worst case.
        possibleGhost = []
        for i in range(len(newGhostStates)):
            ghostX, ghostY = newGhostStates[i].getPosition()
            possibleGhost.extend([(ghostX - 1, ghostY), (ghostX + 1, ghostY),
                                  (ghostX, ghostY + 1), (ghostX, ghostY - 1),
                                  (ghostX, ghostY)])
        if newPos in possibleGhost:
            return float("-inf")

        #we try to prevent pacman from stop action
        if action == Directions.STOP:
            score -= 500

        #We prefer the newposition with food, so it will get bonus.
        for item in currentGameState.getFood().asList():
            if util.manhattanDistance(item, newPos) == 0:
                return float("inf")

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        bestMove, value = self.MiniMax(gameState, 0, self.index)
        return bestMove

    def MiniMax(self, state, depth, index):
        bestMove = None

        role = index % state.getNumAgents() # represents the role of temperary agent, if it equal 0, it is pacman
        if (depth == self.depth and role == 0) or state.isWin() or state.isLose():
            return bestMove, self.evaluationFunction(state)

        #initialized the value.
        if role == 0:
            value = float("-inf")
            newDepth = depth + 1
        else:
            value = float("inf")
            newDepth = depth

        for action in state.getLegalActions(role):
            nextState = state.generateSuccessor(role, action)
            nextMove, nextValue = self.MiniMax(nextState, newDepth, (index + 1))
            if role == 0 and value < nextValue:
                value, bestMove = nextValue, action
            if role != 0 and value > nextValue:
                value, bestMove = nextValue, action

        return bestMove, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestMove, value = self.AlphaBeta(gameState, 0, self.index, float("-inf"), float("inf"))
        return bestMove

    def AlphaBeta(self, state, depth, index, alpha, beta):
        bestMove = None

        role = index % state.getNumAgents() # represents the role of temperary agent, if it equal 0, it is pacman
        if (depth == self.depth and role == 0) or state.isWin() or state.isLose():
            return bestMove, self.evaluationFunction(state)

        #initialized the value.
        if role == 0:
            value = float("-inf")
            newDepth = depth + 1
        else:
            value = float("inf")
            newDepth = depth

        for action in state.getLegalActions(role):
            nextState = state.generateSuccessor(role, action)
            nextMove, nextValue = self.AlphaBeta(nextState, newDepth, (index + 1), alpha, beta)
            if role == 0:
                if value < nextValue:
                    value, bestMove = nextValue, action
                if value >= beta:
                    return bestMove, value
                alpha = max(alpha, value)
            if role != 0:
                if value > nextValue:
                    value, bestMove = nextValue, action
                if value <= alpha:
                    return bestMove, value
                beta = min(beta, value)

        return bestMove, value

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
        bestMove, value = self.Expectimax(gameState, 0, self.index)
        return bestMove

    def Expectimax(self, state, depth, index):
        bestMove = None

        role = index % state.getNumAgents() # represents the role of temperary agent, if it equal 0, it is pacman
        if (depth == self.depth and role == 0) or state.isWin() or state.isLose():
            return bestMove, self.evaluationFunction(state)

        #initialized the value.
        if role == 0:
            value = float("-inf")
            newDepth = depth + 1
        else:
            value = float(0)
            newDepth = depth

        for action in state.getLegalActions(role):
            nextState = state.generateSuccessor(role, action)
            nextMove, nextValue = self.Expectimax(nextState, newDepth, (index + 1))
            if role == 0 and value < nextValue:
                value, bestMove = nextValue, action
            if role != 0:
                #Since we randomly choose an action for ghost, their probability should be same.
                prob = float(1/ len(state.getLegalActions(role)))
                value = float(value + float(prob * nextValue))

        return bestMove, value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    We will calculate the score respect to the closest food distance, the
    minimum distance to the ghost and whether the pacman will eat the food:
    1. we prefer the position with a closer food.
    2. the position of the closest ghost will influence the choice.
    3. if pacman meet the ghost, it will be the worst case, so return infinity
    4. if pacman can eat food at this state, give it a bonus.
    5. with respect to the numver of food left and the scaredtime we have.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #Initialize all score to 0.
    score = 0

    #Find the distance to the closest food
    allFood = newFood.asList()
    if len(allFood) > 0:
        disClosestFood = float("inf")
        for food in allFood:
            tempFoodDis = util.manhattanDistance(newPos, food)
            if tempFoodDis < disClosestFood:
                disClosestFood = tempFoodDis
    else:
        disClosestFood = 0

    #Find the distance to the closest ghost
    disClosestGhost = float("inf")
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        tempGhostDis = util.manhattanDistance(ghostPos, newPos)
        if tempGhostDis < disClosestGhost:
            disClosestGhost = tempGhostDis

    if disClosestGhost <= 0:
        score = float("-inf") #we do not want pacman meet ghost
    else:
        score = score - disClosestFood + 1 / disClosestGhost - 20 * len(allFood) + 0.5 * sum(newScaredTimes)

    #We prefer the newposition with food, so it will get bonus.
    for item in currentGameState.getFood().asList():
        if util.manhattanDistance(item, newPos) == 0:
            return float("inf")

    return score


# Abbreviation
better = betterEvaluationFunction
