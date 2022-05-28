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
        opponent = (agent_id + 1) % 2
        start_time = time.time()
        spare_for_checking = time_limit * 0.5
        legal_ops = env.get_legal_operators(agent_id)
        cur_op = legal_ops[0]
        cur_depth = 0
        while (time.time() - start_time < spare_for_checking) or (cur_depth > env.num_steps):
            legal_ops = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in legal_ops]
            for child, op in zip(children, legal_ops):
                child.apply_operator(agent_id, op)
            possible_moves_for_opp = [self.RB_algo(child, opponent, cur_depth) for child in children]
            best_move_for_opp = max(possible_moves_for_opp)
            cur_op = legal_ops[best_move_for_opp.index()]
            cur_depth += 1
        return cur_op

    def RB_algo(self, env: TaxiEnv, agent_id, depth: int):
        if env.done():
            return env.get_balances()
        if depth == 0:
            return imp_heuristic(env, agent_id)

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
