import sys
import time
from Agent import *
from TaxiEnv import *


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
        return sys.maxsize
    elif diff < 0:
        return -sys.maxsize - 1
    else:
        return 0



def cash_diff(env: TaxiEnv, taxi_id):
    cur_taxi = env.get_taxi(taxi_id)
    opp_taxi = env.get_taxi((taxi_id + 1) % 2)
    return cur_taxi.cash - opp_taxi.cash


class AgentMinimax(Agent):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        start_time = time.time()
        time_to_run_algo = time_limit * 0.1
        depth = 0
        index_selected = 0
        while (time.time() - start_time < time_to_run_algo):
            children_heuristics = [RB_minimax(child, agent_id, depth, True) for child in children]
            depth += 1          
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic) 
        print(operators[index_selected])
        return operators[index_selected]

def RB_minimax(env: TaxiEnv, agent_id: int, depth: int, is_agent_turn: bool):
    if env.done():
        return env.get_balances()
    if depth == 0:
        return imp_heuristic(env, agent_id)
    legal_ops = env.get_legal_operators(agent_id)
    children = [env.clone() for _ in legal_ops]
    if is_agent_turn:
        cur_max = -sys.maxsize - 1
        for child in children:
            v_max = RB_minimax(child, agent_id, depth - 1, False)
            cur_max = max(cur_max, v_max)
        return cur_max
    else: 
        cur_min = sys.maxsize
        for child in children:
            v_max = RB_minimax(child, agent_id, depth - 1, True)
            cur_min = min(cur_min, v_max)
        return cur_min


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
