import sys
import time
from random import randrange

from Agent import *
from TaxiEnv import *

INF = sys.maxsize
M_INF = -sys.maxsize - 1

class AgentGreedyImproved(AgentGreedy):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [imp_heuristic(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


def imp_heuristic(env: TaxiEnv, taxi_id):
    if env.done():
        return decide_by_cash_diff(env, taxi_id)
    cur_taxi = env.get_taxi(taxi_id)
    if env.taxi_is_occupied(taxi_id):
        dist_to_cur_goal = manhattan_distance(cur_taxi.position, cur_taxi.passenger.destination)
    else:
        dist_to_cur_goal = get_closest_pass_plus_dest(env, cur_taxi.position)

    fuel_reward  = cur_taxi.fuel - dist_to_cur_goal
    gas_reward   = get_closest_gas(env, cur_taxi.position) / 2
    cash_reward  = 10 * cash_diff(env, taxi_id)
    imp_huristic = fuel_reward + gas_reward + cash_reward - dist_to_cur_goal
    return imp_huristic


def get_closest_pass_plus_dest(env: TaxiEnv, taxi_pos):
    return min([manhattan_distance(taxi_pos, pas.position) +
                manhattan_distance(pas.position, pas.destination)
                for pas in env.passengers])


def get_closest_gas(env: TaxiEnv, taxi_pos):
    return min([manhattan_distance(taxi_pos, gas.position) for gas in env.gas_stations])


def decide_by_cash_diff(env: TaxiEnv, taxi_id: int):
    diff = cash_diff(env, taxi_id)
    if diff > 0:
        return INF
    elif diff < 0:
        return M_INF
    else:
        return 0


def cash_diff(env: TaxiEnv, taxi_id):
    cur_taxi = env.get_taxi(taxi_id)
    opp_taxi = env.get_taxi((taxi_id + 1) % 2)
    return cur_taxi.cash - opp_taxi.cash


class AgentMinimax(Agent):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        branchFactor = 8
        safetyLimit = 0.1
        depth = 0
        index_selected = 0
        minimax_finish = False
        while ((time.time() - start_time) / time_limit) * branchFactor + safetyLimit <= 1 and not minimax_finish:
            children_heuristics = []
            minimax_finish = True
            for child in children:
                child_heuristics, childFinish = RB_minimax(child, agent_id, depth, True)
                children_heuristics.append(child_heuristics)
                minimax_finish = minimax_finish and childFinish
            depth += 1
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


def RB_minimax(env: TaxiEnv, agent_id: int, depth: int, is_agent_turn: bool):
    if env.done():
        if env.get_balances()[agent_id] - env.get_balances()[1 - agent_id] > 0:
            return INF, True
        else:
            return M_INF, True
    if depth == 0:
        return imp_heuristic(env, agent_id), False
    legal_ops = env.get_legal_operators(agent_id)
    children = [env.clone() for _ in legal_ops]
    if is_agent_turn:
        cur_max = M_INF
        is_finish = True
        for child in children:
            v_max, minimax_finish = RB_minimax(child, agent_id, depth - 1, False)
            cur_max = max(cur_max, v_max)
            is_finish = is_finish and minimax_finish
        return cur_max, is_finish
    else: 
        cur_min = INF
        is_finish = True
        for child in children:
            v_min, minimax_finish = RB_minimax(child, agent_id, depth - 1, True)
            cur_min = min(cur_min, v_min)
            is_finish = is_finish and minimax_finish
        return cur_min, is_finish


class AgentAlphaBeta(Agent):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        branchFactor = 8
        safetyLimit = 0.1
        depth = 0
        index_selected = 0
        minimax_finish = False
        while ((time.time() - start_time) / time_limit) * branchFactor + safetyLimit <= 1 and not minimax_finish:
            children_heuristics = []
            minimax_finish = True
            for child in children:
                child_heuristics, childFinish = RB_minimax_ab(child, agent_id, depth, True, M_INF, INF)
                children_heuristics.append(child_heuristics)
                minimax_finish = minimax_finish and childFinish
            depth += 1
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


def RB_minimax_ab(env: TaxiEnv, agent_id: int, depth: int, is_agent_turn: bool, alpha, beta):
    if env.done():
        if env.get_balances()[agent_id] - env.get_balances()[1 - agent_id] > 0:
            return INF, True
        else:
            return M_INF, True
    if depth == 0:
        return imp_heuristic(env, agent_id), False
    legal_ops = env.get_legal_operators(agent_id)
    children = [env.clone() for _ in legal_ops]
    if is_agent_turn:
        cur_max = M_INF
        is_finish = True
        for child in children:
            v_max, minimax_finish = RB_minimax_ab(child, agent_id, depth - 1, False, alpha, beta)
            cur_max = max(cur_max, v_max)
            alpha = max(cur_max, alpha)
            is_finish = is_finish and minimax_finish
            if cur_max >= beta:
                return INF, is_finish
        return cur_max, is_finish
    else: 
        cur_min = INF
        is_finish = True
        for child in children:
            v_min, minimax_finish = RB_minimax_ab(child, agent_id, depth - 1, True, alpha, beta)
            cur_min = min(cur_min, v_min)
            beta = min(cur_min, beta)
            is_finish = is_finish and minimax_finish
            if cur_min <= alpha:
                return M_INF, is_finish
        return cur_min, is_finish


class AgentExpectimax(Agent):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        branchFactor = 8
        safetyLimit = 0.1
        depth = 0
        index_selected = 0
        minimax_finish = False
        while ((time.time() - start_time) / time_limit) * branchFactor + safetyLimit <= 1 and not minimax_finish:
            children_heuristics = []
            minimax_finish = True
            for child in children:
                child_heuristics, childFinish = RB_expectimax(child, agent_id, depth, True)
                children_heuristics.append(child_heuristics)
                minimax_finish = minimax_finish and childFinish
            depth += 1
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


def RB_expectimax(env: TaxiEnv, agent_id: int, depth: int, is_agent_turn: bool):
    if env.done():
        if env.get_balances()[agent_id] - env.get_balances()[1 - agent_id] > 0:
            return INF, True
        else:
            return M_INF, True
    if depth == 0:
        return imp_heuristic(env, agent_id), False
    operators = env.get_legal_operators(agent_id)
    children = [env.clone() for _ in operators]
    if is_agent_turn:
        cur_max = M_INF
        is_finish = True
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            v_max, expectimax_finish = RB_expectimax(child, agent_id, depth - 1, False)
            cur_max = max(cur_max, v_max)
            is_finish = is_finish and expectimax_finish
        return cur_max, is_finish
    else:
        is_finish = True
        cur_min = INF
        op = get_random_weighted(env, agent_id)
        child = env.clone()
        child.apply_operator(agent_id, op)
        v_min, expectimax_finish = RB_expectimax(child, agent_id, depth - 1, True)
        is_finish = is_finish and expectimax_finish
        cur_min = min(cur_min, v_min)
        return cur_min, is_finish


def get_random_weighted(env: TaxiEnv, agent_id):
    operators = env.get_legal_operators(agent_id)
    weight = []
    for op in operators:
        if op == "refuel" or op == "drop off passenger" or op == "pick up passenger":
            weight.append(2)
        else:
            weight.append(1)
    w_ops = []
    for i, op in enumerate(operators):
        for j in range(weight[i]):
            w_ops.append(op)
    index = randrange(len(w_ops))
    return w_ops[index]

