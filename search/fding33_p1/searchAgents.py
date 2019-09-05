# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.  In this phase, the agent should compute the path to the
        goal and store it in a local variable.  All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        # no other initialization needed
        # a side note on how I designed the states to be
        # a state consists of corresponding position and a tuple to signify if all corners are visited
        # e.g. ((5, 5), (False, False, False, True))
        # Explained: the agent is currently at (5, 5), and it has only visited the top right corner
        "*** MY CODE ENDS HERE ***"

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        "*** YOUR CODE HERE ***"
        return self.startingPosition, (False, False, False, False)
        util.raiseNotDefined()
        "*** MY CODE ENDS HERE ***"

    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        "*** YOUR CODE HERE ***"
        return False not in state[1]
        util.raiseNotDefined()
        "*** MY CODE ENDS HERE ***"

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                next_position = (nextx, nexty)
                next_visited_corners = state[1]
                if next_position in self.corners:
                    next_visited_corners = list(next_visited_corners)
                    next_visited_corners[self.corners.index(next_position)] = True
                    next_visited_corners = tuple(next_visited_corners)
                successor = (next_position, next_visited_corners)
                step_cost = 1
                successors.append((
                    successor,
                    action,
                    step_cost
                ))
            "*** MY CODE ENDS HERE***"

        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    curr_position = state[0]
    unvisited_corners = [corners[i] for i in range(len(corners)) if not state[1][i]]

    heuristic = 0
    while len(unvisited_corners) != 0:
        manhattan_distances = [util.manhattanDistance(curr_position, corner) for corner in unvisited_corners]
        closest_corner_idx = manhattan_distances.index(min(manhattan_distances))
        heuristic = heuristic + manhattan_distances[closest_corner_idx]
        curr_position = unvisited_corners.pop(closest_corner_idx)

    return heuristic
    # return 0 # Default to trivial solution
    "*** MY CODE ENDS HERE***"

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    # 1. this heuristic does not define sub search problem
    # food_list = foodGrid.asList()
    #
    # if len(food_list) == 0:
    #     return 0
    #
    # manhattan_distances = [util.manhattanDistance(position, food_position) for food_position in food_list]
    # closest_food_position = food_list[manhattan_distances.index(min(manhattan_distances))]
    # farthest_food_position = food_list[manhattan_distances.index(max(manhattan_distances))]
    #
    # return util.manhattanDistance(position, closest_food_position) \
    #        + util.manhattanDistance(closest_food_position, farthest_food_position)

    # 2. this heuristic defines a sub search problem to be the actual distance to the farthest food
    def bfs_longest_dist(start, food_grid, total_food_count, wall_grid):

        def successors(curr_pos):
            candidates = [(curr_pos[0], curr_pos[1] + 1),
                          (curr_pos[0] - 1, curr_pos[1]),
                          (curr_pos[0], curr_pos[1] - 1),
                          (curr_pos[0] + 1, curr_pos[1])]
            result = []
            for candidate in candidates:
                if not wall_grid[candidate[0]][candidate[1]]:
                    result.append(candidate)
            return result

        def update_heuristic_info(start, curr_pos, dist):
            problem.heuristicInfo[start][curr_pos] = dist

        visited = set()
        food_count = 0
        queue = util.Queue()
        if start not in problem.heuristicInfo.keys():
            problem.heuristicInfo[start] = {}

        max_dist = -1
        queue.push((start, 0))
        while not queue.isEmpty():
            curr_pos, dist = queue.pop()
            if curr_pos not in visited:
                visited.add(curr_pos)
                # update farthest food distance
                if food_grid[curr_pos[0]][curr_pos[1]]:
                    max_dist = max(dist, max_dist)
                    food_count = food_count + 1
                    update_heuristic_info(start, curr_pos, dist)
                if total_food_count == food_count:
                    return max_dist
                for successor in successors(curr_pos):
                    if successor not in visited:
                        queue.push((successor, dist + 1))

        return max_dist

    total_food_count = len(foodGrid.asList())
    if total_food_count == 0:
        return 0
    else:
        if position in problem.heuristicInfo:
            food_list = foodGrid.asList()
            dist = 0
            for food_pos in food_list:
                if food_pos in problem.heuristicInfo[position]:
                    dist = max(dist, problem.heuristicInfo[position][food_pos])
        else:
            dist = bfs_longest_dist(position, foodGrid, total_food_count, problem.walls)
        return dist

    # return 0
    "*** MY CODE ENDS HERE***"

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.optimalSolutionSizeLimit = 20
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return search.breadthFirstSearch(problem)
        util.raiseNotDefined()
        "*** MY CODE ENDS HERE ***"

class AnyFoodSearchProblem(PositionSearchProblem):
    """
      A search problem for finding a path to any food.

      This search problem is just like the PositionSearchProblem, but
      has a different goal test, which you need to fill in below.  The
      state space and successor function do not need to be changed.

      The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
      inherits the methods of the PositionSearchProblem.

      You can use this search problem to help you fill in
      the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]
        util.raiseNotDefined()
        "*** MY CODE ENDS HERE ***"


##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    # The agent perceives the the food grid as a huge graph, and its job is to visit each node while walking the minimum
    # number of edges (moves). The original problem size is too large for solving even when using Astar with a brilliant
    # heuristic. However, the good news is that this graph is mostly connected and loose, so it is possible to perform a
    # bi-component decomposition on the original graph, where each smaller component can be solved independently for
    # constructing the whole (it can be proved that combining each solution for the bi-connected component always yields
    # the opitional solution for the whole). Therefore, we are solving a lot of tiny-sized problems so that we can run
    # a-star on these small subgraphs and then reconstruct path by joining up them. After this kind of decomposition and
    # dynamic-programming-like procedure, we still have a huge component left: the center, a tough guy with over 100
    # food dots. Instead of directly calling astar search on this graph, the agent simplifies it and only considers the
    # food dots adjacent to at least 3 other food dots. These will be the new vertices, while the others are only
    # considered to be edges. Now the problem is ready to be solved since we have reduced it to P-class (solved in
    # polynomial time): finding a circuit that goes through every edge, kind-of similar to the Chinese Postman Problem.
    # The solution is combined with the ones we previously found to yield the final path.
    # This appoximate search agent should return a path of cost 276 for the big search.
    # A side node: this approach works perfect on large size problems, but it does have one drawback in its assumption:
    # the food grid must be connected, meaning it has only one component

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"

        # import time
        # total_start_time = time.time()

        self.all_actions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST)
        self.direction_shifts = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }
        self.reverse_actions = {
            Directions.NORTH: Directions.SOUTH,
            Directions.SOUTH: Directions.NORTH,
            Directions.EAST: Directions.WEST,
            Directions.WEST: Directions.EAST
        }
        self.visited_sub_solution = set()
        self.curr_sub_solution = []
        pacman_pos = state.getPacmanPosition()
        wall_grid = state.getWalls()
        food_grid = state.getFood()
        # the following line is super super important !!!
        # this makes the maze graph able to be cut-vertex decomposed.
        food_grid[pacman_pos[0]][pacman_pos[1]] = True

        # start_time = time.time()
        cut_vertices = self.get_all_cut_vertices(food_grid)
        # print "decomposition: {} s".format(time.time() - start_time)

        # start_time = time.time()
        self.vertex_solution_returning = self.find_solution_for_cut_vertices(cut_vertices, food_grid, wall_grid, returning=True)
        # print "returning sub solution: {} s".format(time.time() - start_time)

        # start_time = time.time()
        # now we want to run SINGLE TRIP from this greatest_cost_vertex
        vertex_with_greatest_cost = None
        greatest_cost = 0
        for vertex in self.vertex_solution_returning.keys():
            sub_solution = self.vertex_solution_returning[vertex]
            if vertex_with_greatest_cost is None or len(sub_solution) > greatest_cost:
                greatest_cost = len(sub_solution)
                vertex_with_greatest_cost = vertex
        self.critical_vertex = vertex_with_greatest_cost
        # print vertex_with_greatest_cost
        sorted = self.cut_vertex(food_grid, vertex_with_greatest_cost)
        sorted.sort()
        component = sorted[0]
        remaining_vertices = set(cut_vertices.keys()).intersection(set(component[1]))
        remaining_vertices.add(vertex_with_greatest_cost)
        remaining_vertices = {key: cut_vertices[key] for key in remaining_vertices}
        self.vertex_solution_non_returning = self.find_solution_for_cut_vertices(remaining_vertices, food_grid, wall_grid, returning=False)
        # print "non returning sub solution: {} s".format(time.time() - start_time)

        # print "run time: {}s".format(time.time() - total_start_time)

        # start_time = time.time()
        # remaining dots to eat after decomposition
        modified_food_grid = self.remove_sub_solutions(food_grid, cut_vertices)
        adj_list = self.generate_graph(modified_food_grid, pacman_pos, vertex_with_greatest_cost)
        # self.star_print(modified_food_grid, pacman_pos)

        adj_list_keys = list(adj_list.keys())
        adj_list_keys.sort(key=lambda vertex: len(adj_list[vertex][0][1]))

        # swap the first with the non-returning vertex
        idx = adj_list_keys.index(vertex_with_greatest_cost)
        temp = adj_list_keys[idx]
        adj_list_keys[idx] = adj_list_keys[0]
        adj_list_keys[0] = temp

        visited = set()
        for vertex in adj_list_keys:
            visited.add(vertex)
            if len(adj_list[vertex]) > 2 or vertex == vertex_with_greatest_cost:
                for i in range(len(adj_list[vertex])):
                    repeated_vertex, actions = adj_list[vertex][i]
                    if len(adj_list[repeated_vertex]) > 2 or repeated_vertex == pacman_pos:
                        adj_list[vertex][i] = (vertex, self.cut_last_reverse_and_invert(actions))
                        if repeated_vertex not in visited or repeated_vertex == pacman_pos:
                            self.remove_from_adj_list(adj_list, repeated_vertex, vertex)
                        break

        # for vertex in adj_list_keys:
        #     print vertex
        #     for next_vertex, actions in adj_list[vertex]:
        #         print next_vertex, actions
        #     print " "

        self.main_actions = []
        visited = set()
        curr_pos = pacman_pos
        while len(visited) < len(adj_list_keys):
            visited.add(curr_pos)
            for next_vertex, actions in adj_list[curr_pos]:
                if next_vertex == curr_pos:
                    self.main_actions = self.main_actions + actions
                    break
            for next_vertex, actions in adj_list[curr_pos]:
                if next_vertex not in visited:
                    self.main_actions = self.main_actions + actions
                    curr_pos = next_vertex
        "*** MY CODE ENDS HERE ***"

    def getAction(self, state):
        """
        From game.py:
        The Agent will receive a GameState and must return an action from
        Directions.{North, South, East, West, Stop}
        """
        "*** YOUR CODE HERE ***"
        curr_pos = state.getPacmanPosition()

        # print curr_pos
        # print(self.curr_sub_solution)

        if len(self.curr_sub_solution) == 0:
            if curr_pos in self.vertex_solution_non_returning:
                if curr_pos not in self.visited_sub_solution:
                    self.curr_sub_solution = self.curr_sub_solution + self.vertex_solution_non_returning[curr_pos]
                    self.visited_sub_solution.add(curr_pos)
            elif curr_pos in self.vertex_solution_returning:
                if curr_pos not in self.visited_sub_solution:
                    self.curr_sub_solution = self.curr_sub_solution + self.vertex_solution_returning[curr_pos]
                    self.visited_sub_solution.add(curr_pos)

        if self.critical_vertex in self.visited_sub_solution and len(self.main_actions) > 0:
            return self.main_actions.pop(0)

        if len(self.curr_sub_solution) > 0:
            return self.curr_sub_solution.pop(0)

        if len(self.main_actions) > 0:
            return self.main_actions.pop(0)
        else:
            return Directions.STOP

        util.raiseNotDefined()
        "*** MY CODE ENDS HERE ***"

    "*** Some other helper functions ***"
    # # used for debugging
    # def star_print(self, food_grid, curr_pos):
    #     for y in range(food_grid.height)[::-1]:
    #         row = ""
    #         for x in range(food_grid.width):
    #             if (x, y) == curr_pos:
    #                 row = row + "*"
    #             elif food_grid[x][y]:
    #                 row = row + "o"
    #             else:
    #                 row = row + "+"
    #         print row

    def find_solution_for_cut_vertices(self, cut_vertices, food_grid, wall_grid, returning):
        start_time = time.time()
        for cut_vertex in cut_vertices.keys():
            cut_vertices[cut_vertex].sort()
        sorted_cut_vertices_keys = self.sorted_cut_vertices_keys(cut_vertices)
        # print "sorting: {} s".format(time.time() - start_time)

        # this is used for memorizing solutions for cut vertex positions
        vertex_solution = {}
        for idx in range(len(sorted_cut_vertices_keys)):
            start_vertex = sorted_cut_vertices_keys[idx]
            components = cut_vertices[sorted_cut_vertices_keys[idx]]
            # always solve the smallest component problem
            # exception: when the cut vertex divides graph into 3 components
            smallest_component = components[0]
            component_size, food_to_visit, allowed_actions = smallest_component

            # here becomes interesting: we are using previous solutions to build this solution.
            food_to_visit = set(food_to_visit)
            min_prob_size = len(food_to_visit)
            best_jdx = None
            for jdx in range(idx):
                food_previously_visited = set(cut_vertices[sorted_cut_vertices_keys[jdx]][0][1])
                remain_food_to_visit = food_to_visit - food_previously_visited
                if len(remain_food_to_visit) < min_prob_size:
                    min_prob_size = len(remain_food_to_visit)
                    best_jdx = jdx
            if best_jdx is not None:
                food_to_visit = food_to_visit - set(cut_vertices[sorted_cut_vertices_keys[best_jdx]][0][1])

            modified_food_grid = food_grid.copy()
            for x in range(modified_food_grid.width):
                for y in range(modified_food_grid.height):
                    if (x, y) not in food_to_visit:
                        modified_food_grid[x][y] = False

            forbidden_actions = set(self.all_actions) - set(allowed_actions)

            search_prob = ReturnFoodSearchProblem(start_vertex, wall_grid, modified_food_grid, forbidden_actions,
                                                  returning=returning)

            # start_time = time.time()
            # actions_ucs = search.ucs(search_prob)
            # print "ucs: {} s".format(time.time() - start_time)

            start_time = time.time()
            actions_astar = search.astar(search_prob, chamber_heuristic)

            # print sorted_cut_vertices_keys[idx], len(food_to_visit)
            # print "astar: {} s".format(time.time() - start_time)
            # print "astar: {}".format(actions_astar)

            # if this solution is built upon previous solutions, we need to reconstruct!
            if best_jdx is not None:
                sub_vertex = sorted_cut_vertices_keys[best_jdx]
                sub_solution = vertex_solution[sub_vertex]
                if returning:
                    actions_astar = self.reconstruct_path_returning(actions_astar, sub_solution, start_vertex, sub_vertex)
                else:
                    actions_astar = self.reconstruct_path_single(actions_astar, sub_solution, start_vertex, sub_vertex)
                # print("this solution uses sub solution: {}".format(sub_solution))

            # print "reconstruct_astar actions: {}".format(actions_astar)

            # the solution should be memorized for future reference
            vertex_solution[sorted_cut_vertices_keys[idx]] = actions_astar

            # assert len(actions_ucs) == len(actions_astar)

            # print "ucs: {}".format(actions_ucs)
            # print "astar: {}".format(actions_astar)

        return vertex_solution


    def get_all_cut_vertices(self, food_grid):
        cut_vertices = {}
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:
                    components = self.cut_vertex(food_grid, (x, y))
                    if len(components) > 1:
                        cut_vertices[(x, y)] = components
        return cut_vertices

    # for a given vertex, the function returns how this vertex divides the original food grid as a graph;
    # a cut vertex increases the component count of the original graph. The method returns a list of triples;
    # each triple contains the size of each component, the vertices in that component, and what actions to take from
    # the cut vertex into that component
    def cut_vertex(self, food_grid, vertex):
        components = []
        visited = set()
        dx = (0, 0, 1, -1)
        dy = (1, -1, 0, 0)
        reverse_actions = {Directions.NORTH: Directions.SOUTH,
                           Directions.SOUTH: Directions.NORTH,
                           Directions.EAST: Directions.WEST,
                           Directions.WEST: Directions.EAST}
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y] and (x, y) != vertex and (x, y) not in visited:
                    queue = util.Queue()
                    queue.push(((x, y), None))
                    curr_size = 0
                    how_into = []
                    component_vertices = []
                    while not queue.isEmpty():
                        curr_pos, prev_action = queue.pop()
                        if curr_pos == vertex:
                            continue
                        if curr_pos not in visited:
                            visited.add(curr_pos)
                            curr_size = curr_size + 1
                            component_vertices.append(curr_pos)
                            for i in range(len(dx)):
                                next_x, next_y = curr_pos[0] + dx[i], curr_pos[1] + dy[i]
                                next_pos = (next_x, next_y)
                                if next_pos == vertex:
                                    how_into.append(self.reverse_actions[self.all_actions[i]])
                                if food_grid[next_x][next_y] and next_pos not in visited:
                                    queue.push((next_pos, self.all_actions[i]))
                    components.append((curr_size, component_vertices, how_into))
        return components

    def sorted_cut_vertices_keys(self, cut_vertices):
        size_vertex = []
        for cut_vertex in cut_vertices.keys():
            size_vertex.append((cut_vertices[cut_vertex][0][0], cut_vertex))
        size_vertex.sort()
        return [vertex for _, vertex in size_vertex]

    def reconstruct_path_returning(self, curr_solution, sub_solution, start_vertex, sub_vertex):
        curr_vertex = start_vertex
        insert_idx = 0
        while curr_vertex != sub_vertex:
            shift = self.direction_shifts[curr_solution[insert_idx]]
            curr_vertex = (curr_vertex[0] + shift[0], curr_vertex[1] + shift[1])
            insert_idx = insert_idx + 1
        return curr_solution[:insert_idx] + sub_solution + curr_solution[insert_idx:]

    def reconstruct_path_single(self, curr_solution, sub_solution, start_vertex, sub_vertex):
        curr_vertex = start_vertex
        split_idx = 0
        while curr_vertex != sub_vertex:
            shift = self.direction_shifts[curr_solution[split_idx]]
            curr_vertex = (curr_vertex[0] + shift[0], curr_vertex[1] + shift[1])
            split_idx = split_idx + 1

        if split_idx > 0 and split_idx < len(curr_solution) \
                and self.reverse_actions[curr_solution[split_idx - 1]] == curr_solution[split_idx]:
            return curr_solution[:split_idx - 1] + curr_solution[split_idx + 1:] \
                   + [self.reverse_actions[each_direction] for each_direction in curr_solution[split_idx:][::-1]] \
                   + sub_solution
        else:
            return curr_solution \
                   + [self.reverse_actions[each_direction] for each_direction in curr_solution[split_idx:][::-1]] \
                   + sub_solution

    def remove_sub_solutions(self, food_grid, cut_vertices):
        all_food = set(food_grid.asList())
        for vertex in cut_vertices.keys():
            all_food = all_food - set(cut_vertices[vertex][0][1])
        modified_food_grid = food_grid.copy()
        for x in range(modified_food_grid.width):
            for y in range(modified_food_grid.height):
                modified_food_grid[x][y] = False
        for food in all_food:
            modified_food_grid[food[0]][food[1]] = True
        return modified_food_grid

    def generate_graph(self, food_grid, pacman_pos, end_pos):
        adj_set = set()
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:
                    adj_count = 0
                    for shift in [self.direction_shifts[direction] for direction in self.all_actions]:
                        if food_grid[x + shift[0]][y + shift[1]]:
                            adj_count = adj_count + 1
                    if adj_count > 2:
                        adj_set.add((x, y))
        adj_set.add(pacman_pos)
        adj_set.add(end_pos)
        adj_list = {}
        for vertex in adj_set:
            adj_list[vertex] = self.bfs_adj_vertex(food_grid, vertex, adj_set)
        return adj_list

    def bfs_adj_vertex(self, food_grid, start_vertex, adj_set):
        res = []
        queue = util.Queue()
        visited = set()
        queue.push((start_vertex, []))
        while not queue.isEmpty():
            curr_vertex, actions = queue.pop()
            if curr_vertex != start_vertex and curr_vertex in adj_set:
                res.append((curr_vertex, actions))
                continue
            if curr_vertex not in visited:
                visited.add(curr_vertex)
                for direction in self.all_actions:
                    shift = self.direction_shifts[direction]
                    next_x, next_y = curr_vertex[0] + shift[0], curr_vertex[1] + shift[1]
                    if food_grid[next_x][next_y] and (next_x, next_y) not in visited:
                        queue.push(((next_x, next_y), actions + [direction]))
        return res

    def cut_last_reverse_and_invert(self, actions):
        actions = actions[:len(actions) - 1]
        return actions + [self.reverse_actions[action] for action in actions[::-1]]

    def remove_from_adj_list(self, adj_list, from_vertex, to_vertex):
        next_vertices = adj_list[from_vertex]
        for i in range(len(next_vertices)):
            if next_vertices[i][0] == to_vertex:
                next_vertices.pop(i)
                return
    "*** End of all my helper functions***"


"*** Some new serach problems that I define ***"
class ReturnFoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) AND returns to the original position in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startPosition, wallGrid, foodGrid, forbiddenStartActions, returning):
        self.startState = (startPosition, foodGrid)
        self.walls = wallGrid
        self.forbiddenStartActions = forbiddenStartActions
        self.returning = returning
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state[1].count() == 0 and (not self.returning or state[0] == self.startState[0])

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            if state[0] == self.startState[0] and direction in self.forbiddenStartActions:
                continue
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_food = state[1].copy()
                next_food[nextx][nexty] = False
                successors.append((((nextx, nexty), next_food), direction, 1))
        return successors

    def getGoalPosition(self):
        return self.startState[0]
"*** End of search problem definition ***"

"*** Some new heuristic I created for eating large chunks of food ***"
def chamber_heuristic(state, problem):
    position, foodGrid = state
    goal_position = problem.getGoalPosition()
    food_list = foodGrid.asList()
    if len(food_list) == 0:
        if problem.returning:
            return util.manhattanDistance(position, goal_position)
        else:
            return 0
    else:
        if problem.returning:
            return closest_manhattan_dist(problem, position, food_list) + len(food_list) + closest_manhattan_dist(problem, goal_position, food_list)
        else:
            return closest_manhattan_dist(problem, position, food_list) + len(food_list)


def closest_manhattan_dist(problem, position, food_list):
    min_dist = 999999
    for food in food_list:
        min_dist = min(query_manhattan_distance(problem, position, food), min_dist)
    return min_dist


def query_manhattan_distance(problem, position1, position2):
    if (position1, position2) not in problem.heuristicInfo and (position2, position1) not in problem.heuristicInfo:
        problem.heuristicInfo[(position1, position2)] = util.manhattanDistance(position1, position2)
    if (position2, position1) not in problem.heuristicInfo:
        problem.heuristicInfo[(position2, position1)] = problem.heuristicInfo[(position1, position2)]
    return problem.heuristicInfo[(position1, position2)]
"*** End of my heuristic definition ***"


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
