# -*- coding: utf-8 -*-




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

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    ############################################################################
    # Task1: Finding a Fixed Food Dot using Depth First Search #

    from util import Stack

    # 스택 초기화: 시작 상태와 빈 경로 목록을 포함하는 튜플을 스택에 푸시
    frontier = Stack()
    frontier.push((problem.getStartState(), []))
    
    # 방문한 노드를 추적하는 집합
    visited = set()

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        # 현재 상태가 목표 상태인지 확인
        if problem.isGoalState(current_state):
            return path
        
        # 현재 상태를 방문 처리
        if current_state not in visited:
            visited.add(current_state)

            # 후속 상태를 스택에 추가
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:

                    # 현재 경로에 새로운 행동을 추가하여 새로운 경로 생성
                    new_path = path + [action]
                    frontier.push((successor, new_path))

    # 찾지 못하면 빈 리스트를 반환합니다.
    return []

########################################################################
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()






def breadthFirstSearch(problem):

    """Search the shallowest nodes in the search tree first."""
    #########################################################################
    # Task2: Breadth First Search


    from util import Queue
     # 큐를 초기화합니다. 시작 상태와 빈 경로, 그리고 초기 비용 0을 함께 큐에 추가합니다.
    queue = Queue()
    visited = set()  # 방문한 노드를 추적하기 위한 집합입니다.
    start_state = problem.getStartState()
    queue.push((start_state, [], 0))  # 시작 상태와 빈 경로, 비용을 큐에 푸시합니다.

    # 큐가 빌 때까지 계속 반복합니다.
    while not queue.isEmpty():
        current_state, actions, cost = queue.pop()  # 큐에서 현재 상태, 행동 목록, 비용을 추출합니다.

        # 현재 상태가 목표 상태인지 확인합니다.
        if problem.isGoalState(current_state):
            return actions  # 목표 상태에 도달하면 행동 목록을 반환합니다.

        # 현재 상태를 방문한 적이 없다면 방문 처리를 하고 후속 상태를 큐에 추가합니다.
        if current_state not in visited:
            visited.add(current_state)  # 현재 상태를 방문한 상태로 추가합니다.

            # 현재 상태의 모든 후속 상태를 가져와 큐에 추가합니다.
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:  # 아직 방문하지 않은 후속 상태만 처리합니다.
                    new_actions = actions + [action]  # 현재 행동 목록에 새 행동을 추가합니다.

                    # 후속 상태, 새로운 행동 목록, 누적 비용을 큐에 푸시합니다.
                    queue.push((successor, new_actions, cost + step_cost))

    # 찾지 못하면 빈 리스트를 반환합니다.
    return []
    #################################################################################
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()






def uniformCostSearch(problem):
    
    """Search the node of least total cost first."""
    ##################################################################
    ## ## Task3: Varying the Cost Function
    """노드의 총 비용이 가장 낮은 순서대로 검색합니다."""
    from util import PriorityQueue
    pqueue = PriorityQueue()  # 우선순위 큐 초기화
    visited = set()  # 방문한 노드를 추적하기 위한 집합
    start_state = problem.getStartState()
    pqueue.push((start_state, [], 0), 0)  # 시작 상태와 빈 경로, 초기 비용을 큐에 추가

    while not pqueue.isEmpty():
        current_state, actions, cost = pqueue.pop()  # 큐에서 현재 상태, 행동 목록, 비용을 추출

        # 현재 상태가 목표 상태인지 확인
        if problem.isGoalState(current_state):
            return actions

        # 현재 상태를 방문한 적이 없다면 후속 상태로 확장
        if current_state not in visited:
            visited.add(current_state)  # 현재 상태를 방문한 상태로 추가

            # 현재 상태의 모든 후속 상태를 처리
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:  # 아직 방문하지 않은 후속 상태만 처리
                    new_cost = cost + step_cost  # 새로운 총 비용 계산
                    new_actions = actions + [action]  # 행동 목록 업데이트
                    # 우선순위 큐에 후속 상태를 추가하거나 기존 노드를 업데이트
                    pqueue.update((successor, new_actions, new_cost), new_cost)

    return []
#########################################################################################
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #########################################################################
    from util import PriorityQueue
    pqueue = PriorityQueue()
    visited = set()
    start = problem.getStartState()
    pqueue.push((start, [], 0), heuristic(start, problem))

    while not pqueue.isEmpty():
        current_state, actions, cost = pqueue.pop()
        if problem.isGoalState(current_state):
            return actions
        if current_state not in visited:
            visited.add(current_state)
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = cost + step_cost
                new_actions = actions + [action]
                heuristic_cost = new_cost + heuristic(successor, problem)
                pqueue.update((successor, new_actions, new_cost), heuristic_cost)
    return []
###################################################################################
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
