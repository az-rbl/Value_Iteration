"""Pruebas unitarias e integración para value.py en Taxi-v4."""

import gymnasium as gym
import numpy as np

from value import (
    ENV_NAME,
    EPSILON,
    evaluate_policy_episodes,
    evaluate_policy_q,
    make_biased_movement_policy,
    make_epsilon_greedy_policy,
    make_fixed_action_policy,
    make_random_deterministic_policy,
    make_uniform_policy,
    policy_iteration,
    q_value_iteration,
)


def test_policies_are_valid_distributions():
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_dummy = np.zeros((n_states, n_actions), dtype=np.float64)
    policies = [
        make_uniform_policy(n_states, n_actions),
        make_biased_movement_policy(n_states, n_actions),
        make_fixed_action_policy(n_states, n_actions, action=0),
        make_random_deterministic_policy(n_states, n_actions),
        make_epsilon_greedy_policy(q_dummy, epsilon=0.2),
    ]
    for policy in policies:
        assert policy.shape == (n_states, n_actions)
        assert np.all(policy >= 0.0)
        assert np.allclose(np.sum(policy, axis=1), 1.0)
    env.close()


def test_fixed_policy_evaluation_converges_and_ranks_correctly():
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    res_uniform = evaluate_policy_q(env, make_uniform_policy(n_states, n_actions))
    res_biased = evaluate_policy_q(env, make_biased_movement_policy(n_states, n_actions))

    assert res_uniform.converged
    assert res_biased.converged
    assert res_uniform.deltas[-1] < EPSILON
    assert res_biased.deltas[-1] < EPSILON
    # Evitar acciones ilegales frecuentes mejora Q promedio respecto a uniforme
    assert np.mean(res_biased.q_table) > np.mean(res_uniform.q_table)
    env.close()


def test_policy_iteration_and_q_value_iteration_converge_to_same_q_star():
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    qvi_zero = q_value_iteration(env)
    qvi_opt = q_value_iteration(env, initial_q=np.full((n_states, n_actions), 20.0))
    qvi_pess = q_value_iteration(env, initial_q=np.full((n_states, n_actions), -20.0))

    pi_uniform = policy_iteration(env, make_uniform_policy(n_states, n_actions))
    pi_south = policy_iteration(env, make_fixed_action_policy(n_states, n_actions, 0))

    for res in (qvi_zero, qvi_opt, qvi_pess, pi_uniform, pi_south):
        assert res.converged
        assert res.deltas[-1] < EPSILON
        assert np.max(np.abs(res.q_table - qvi_zero.q_table)) < 1e-4

    env.close()


def test_optimal_policy_solves_taxi_v4():
    env = gym.make(ENV_NAME)
    qvi_res = q_value_iteration(env)
    metrics = evaluate_policy_episodes(env, qvi_res.policy, n_episodes=50, seed=123)

    assert metrics.mean_reward > 7.0
    assert metrics.success_rate == 1.0
    env.close()
