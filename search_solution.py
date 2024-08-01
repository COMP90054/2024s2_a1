# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Grid
from searchProblems import nullHeuristic,PositionSearchProblem,ConstrainedAstarProblem

### You might need to use
from copy import deepcopy







def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    Q = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    Q.push(startNode)
    while not Q.isEmpty():
        node = Q.pop()
        state, cost, path = node
        if problem.isGoalState(state):
            return path
        for succ in problem.getSuccessors(state):
            succState, succAction, succCost = succ
            new_cost = cost + succCost
            newNode = (succState, new_cost, path + [succAction])
            Q.push(newNode)

    return None  # Goal not found


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



def goalCountingHeuristic(state, problem):
    return state[1].count()


def minFoodHeuristic(state, problem):
    foodGrid = state[1]
    pos = state[0]
    if foodGrid.count() == 0:
        return 0

    min_dis = 999999
    for food_pos in foodGrid.asList():
        min_dis = min(util.manhattanDistance(food_pos, pos), min_dis)
    return max(min_dis, len(state[1].asList()))


def maxFoodHeuristic(state, problem):
    foodGrid = state[1]
    pos = state[0]
    max_dis = 0
    for food_pos in foodGrid.asList():
        max_dis = max(util.manhattanDistance(food_pos, pos), max_dis)
    return max(max_dis, len(state[1].asList()))

def maxMazeFoodHeuristic(state, problem):
    foodGrid = state[1]
    pos = state[0]
    max_dis = 0
    for food_pos in foodGrid.asList():
        max_dis = max(mazeDistance(food_pos, pos, problem), max_dis)
    return max(max_dis, len(state[1].asList()))

def mazeDistance(point1, point2, problem):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    if (x1, y1, x2, y2) in problem.heuristicInfo:
        return problem.heuristicInfo[(x1, y1, x2, y2)]

    gameState = problem.startingGameState
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    dis = len(astar(prob,positionSearchHeuristic))
    problem.heuristicInfo[(x1, y1, x2, y2)] = dis
    return dis

def positionSearchHeuristic(state, problem):
    """
    Heuristic function used by constrained astar search.
    """
    return util.manhattanDistance(state, problem.goal)

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for task1 ***"
    # return maxFoodHeuristic(state, problem)
    return maxMazeFoodHeuristic(state, problem)

def CAstarHeuristic(state, problem):
    pos = state[0]
    assert state[1].count() <= 1
    if state[1].count() == 0:
        return 0

    food_pos = state[1].asList()[0]
    return util.manhattanDistance(food_pos, pos)


from searchProblems import MAPFProblem
def aStarSearchConstraints(problem, vertex_constraints=[], edge_constraints=[],heuristic=CAstarHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [startState], [])
    myPQ.push(startNode, heuristic(startState, problem))
    closed = set([c for c in vertex_constraints])
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, trajs, path = node
        if not (state[0], cost) in closed:
            closed.add((state[0], cost))
            if problem.isGoalState(state):
                for (pos,
                     t) in vertex_constraints:  # if there is a constraint that the pacman can not be in the current position at time t,
                    if state[0] == pos and cost <= t:  # then it is not a goal state
                        break
                else:
                    return trajs, path

            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                if not (state[0], succState[0], cost) in edge_constraints:
                    new_cost = cost + succCost
                    newNode = (succState, new_cost, trajs + [succState], path + [succAction])
                    myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None, None  # Goal not found




def conflictBasedSearch(problem: MAPFProblem):
    """
        Conflict-based search algorithm.
        Input: MAPFProblem
        Output(IMPORTANT!!!): A dictionary stores path for each pacman as a list {pacman_name: [a1, a2, ...]}.

        A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
          pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
          foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
        
        Hints:
            You should model the constrained Astar problem as a food search problem instead of a position search problem
    """
    "*** YOUR CODE HERE for task3 ***"
    pacman_positions, food_grid = problem.getStartState()
    v_constraints_dic = {p: [] for p in pacman_positions.keys()}
    e_constraints_dic = {p: [] for p in pacman_positions.keys()}
    trajs = {} # each item is a tuple of (trajs, path)
    walls = problem.walls
    probs = {}
    for pacman in pacman_positions.keys():
        x1, y1 = pacman_positions[pacman]
        x2, y2 = food_grid.asList(pacman)[0]
        new_food_grid = Grid(walls.width, walls.height)
        new_food_grid[x2][y2] = True
        probs[pacman] = ConstrainedAstarProblem(pos=(x1, y1), food=new_food_grid, walls=walls)
        trajs[pacman] = aStarSearchConstraints(probs[pacman])

    start_node = (trajs, v_constraints_dic, e_constraints_dic)
    sum_cost = sum([len(trajs[p]) - 1 for p in pacman_positions.keys()])
    pq = util.PriorityQueue()
    pq.push(start_node, sum_cost)

    while not pq.isEmpty():
        s = pq.getMinimumPriority()
        ts, vcs, ecs = pq.pop()
        c = validate({p:ts[p][0] for p in ts})  # c = (p_i, p_j, pos, t) or (p_i, p_j, pos1, pos2, t)
        if not c:
            # print("path found")
            # print({p:ts[p][1] for p in ts})
            return {p:ts[p][1] for p in ts} # only return dictionary of path

        if len(c) == 4:
            p_i, p_j, pos, t = c

            new_constraints = deepcopy(vcs)
            new_constraints[p_i].append((pos, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_i] = aStarSearchConstraints(probs[p_i], new_constraints[p_i], ecs[p_i])
            if new_trajs[p_i] != (None, None):
                pq.push((new_trajs, new_constraints, ecs), s - len(ts[p_i][0]) + len(new_trajs[p_i][0]))

            new_constraints = deepcopy(vcs)
            new_constraints[p_j].append((pos, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_j] = aStarSearchConstraints(probs[p_j], new_constraints[p_j], ecs[p_j])
            if new_trajs[p_j] != (None, None):
                pq.push((new_trajs, new_constraints, ecs), s - len(ts[p_j][0]) + len(new_trajs[p_j][0]))
        else:  # len(c) == 5, edge conflict
            p_i, p_j, pos1, pos2, t = c
            new_constraints = deepcopy(ecs)
            new_constraints[p_i].append((pos1, pos2, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_i] = aStarSearchConstraints(probs[p_i], vcs[p_i], new_constraints[p_i])
            if new_trajs[p_i] != (None, None):
                pq.push((new_trajs, vcs, new_constraints), s - len(ts[p_i][0]) + len(new_trajs[p_i][0]))

            new_constraints = deepcopy(ecs)
            new_constraints[p_j].append((pos2, pos1, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_j] = aStarSearchConstraints(probs[p_j], vcs[p_j], new_constraints[p_j])
            if new_trajs[p_j] != (None, None):
                pq.push((new_trajs, vcs, new_constraints), s - len(ts[p_j][0]) + len(new_trajs[p_j][0]))

    return None


def validate(trajs):
    # vertex conflict
    t = 1
    flag = True

    while flag:
        poss = {}
        flag = False
        for p in trajs:
            if len(trajs[p]) > t:
                pos = trajs[p][t][0]
                flag = True
            else:
                pos = trajs[p][-1][0]

            if pos in poss:
                return p, poss[pos], pos, t
            poss[pos] = p

        t += 1

    # edge conflict
    t = 1
    flag = True

    while flag:
        edges = {}
        flag = False
        for p in trajs:
            if len(trajs[p]) > t:
                pre_pos = trajs[p][t - 1][0]
                pos = trajs[p][t][0]
                if (pos, pre_pos) in edges:
                    return p, edges[(pos, pre_pos)], pre_pos, pos, t - 1
                edges[(pre_pos, pos)] = p
                flag = True

        t += 1

    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch
