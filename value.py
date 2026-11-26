#!/usr/bin/env python3
"""
Iteración de Políticas y Valor para la función Q(s, a) en Taxi-v4 (Gymnasium).

Implementa:
1. Evaluación iterativa de políticas para estimar Q^pi(s, a) dada una política fija pi.
2. Iteración de Políticas (Policy Iteration) partiendo de distintas políticas iniciales.
3. Iteración del Valor para Q (Q-Value Iteration) desde distintas inicializaciones Q_0.
4. Criterio formal de convergencia (delta = max |Q_{k+1} - Q_k| < epsilon).
5. Evaluación de políticas mediante interacción con la API de Gymnasium (reset / step),
   registro en TensorBoard, gráficas PNG con Matplotlib y resumen JSON.
"""

from __future__ import annotations

import json
import typing as tt
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

ENV_NAME = "Taxi-v4"
GAMMA = 0.9
EPSILON = 1e-6
MAX_ITERATIONS = 10_000
TEST_EPISODES = 100
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = BASE_DIR / "runs"


@dataclass
class EvalMetrics:
    """Métricas de desempeño obtenidas al ejecutar episodios en Gymnasium."""

    mean_reward: float
    std_reward: float
    mean_steps: float
    success_rate: float


@dataclass
class ConvergenceResult:
    """Resultado completo de un experimento de estimación o iteración de Q."""

    name: str
    q_table: np.ndarray
    policy: np.ndarray
    deltas: list[float]
    iterations: int
    outer_iterations: int
    converged: bool
    eval_metrics: EvalMetrics | None = None


def validate_policy(policy: np.ndarray, n_states: int, n_actions: int) -> None:
    """Verifica que la matriz de política tenga forma (S, A) y filas estocásticas."""
    if policy.shape != (n_states, n_actions):
        raise ValueError(
            f"Forma de política inválida: {policy.shape}, se esperaba {(n_states, n_actions)}"
        )
    if np.any(policy < 0.0):
        raise ValueError("La política no puede contener probabilidades negativas.")
    row_sums = np.sum(policy, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Cada fila de la política debe sumar 1.0.")


def make_uniform_policy(n_states: int, n_actions: int) -> np.ndarray:
    """Política aleatoria uniforme: pi(a|s) = 1 / |A|."""
    return np.full((n_states, n_actions), 1.0 / n_actions, dtype=np.float64)


def make_biased_movement_policy(n_states: int, n_actions: int) -> np.ndarray:
    """
    Política estocástica sesgada hacia acciones de desplazamiento.

    En Taxi-v4 las acciones 0..3 son Sur, Norte, Este, Oeste y 4..5 son
    Pickup y Dropoff (las cuales penalizan -10 si se ejecutan ilegalmente).
    """
    if n_actions != 6:
        return make_uniform_policy(n_states, n_actions)
    probs = np.array([0.225, 0.225, 0.225, 0.225, 0.05, 0.05], dtype=np.float64)
    return np.tile(probs, (n_states, 1))


def make_fixed_action_policy(
    n_states: int, n_actions: int, action: int = 0
) -> np.ndarray:
    """Política determinista que siempre elige la misma acción (por defecto Sur=0)."""
    policy = np.zeros((n_states, n_actions), dtype=np.float64)
    policy[:, action] = 1.0
    return policy


def make_random_deterministic_policy(
    n_states: int, n_actions: int, seed: int = SEED
) -> np.ndarray:
    """Política determinista aleatoria fija generada con semilla reproducible."""
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, n_actions, size=n_states)
    policy = np.zeros((n_states, n_actions), dtype=np.float64)
    policy[np.arange(n_states), actions] = 1.0
    return policy


def greedy_policy_from_q(q_table: np.ndarray) -> np.ndarray:
    """Extrae la política determinista greedy pi(s) = argmax_a Q(s, a)."""
    n_states, n_actions = q_table.shape
    best_actions = np.argmax(q_table, axis=1)
    policy = np.zeros((n_states, n_actions), dtype=np.float64)
    policy[np.arange(n_states), best_actions] = 1.0
    return policy


def make_epsilon_greedy_policy(
    q_table: np.ndarray, epsilon: float = 0.2
) -> np.ndarray:
    """Construye una política epsilon-greedy a partir de una tabla Q."""
    n_states, n_actions = q_table.shape
    policy = np.full((n_states, n_actions), epsilon / n_actions, dtype=np.float64)
    best_actions = np.argmax(q_table, axis=1)
    policy[np.arange(n_states), best_actions] += 1.0 - epsilon
    return policy


def _bellman_q_sweep(
    transitions: tt.Mapping[int, tt.Mapping[int, list[tuple[float, int, float, bool]]]],
    n_states: int,
    n_actions: int,
    next_state_values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Ejecuta un barrido síncrono de Bellman sobre todos los pares (s, a).

    Q_new(s, a) = sum_{s'} P(s'|s, a) * [r + gamma * (1 - done) * V(s')]
    donde V(s') proviene de evaluar la política (esperanza) o de max_a' Q(s', a').
    """
    q_new = np.zeros((n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        state_transitions = transitions[state]
        for action in range(n_actions):
            expected_q = 0.0
            for prob, next_state, reward, terminated in state_transitions[action]:
                future_val = 0.0 if terminated else next_state_values[next_state]
                expected_q += prob * (float(reward) + gamma * future_val)
            q_new[state, action] = expected_q
    return q_new


def evaluate_policy_q(
    env: gym.Env,
    policy: np.ndarray,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
    max_iterations: int = MAX_ITERATIONS,
    initial_q: np.ndarray | None = None,
    name: str = "policy_eval",
) -> ConvergenceResult:
    """
    Estima Q^pi(s, a) para una política fija pi mediante evaluación iterativa.

    Criterio de convergencia: max_{s, a} |Q_{k+1}(s, a) - Q_k(s, a)| < epsilon.
    """
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    validate_policy(policy, n_states, n_actions)
    transitions = env.unwrapped.P

    q_table = (
        np.zeros((n_states, n_actions), dtype=np.float64)
        if initial_q is None
        else initial_q.astype(np.float64, copy=True)
    )
    deltas: list[float] = []
    converged = False

    for _ in range(max_iterations):
        # V^pi(s') = sum_{a'} pi(a'|s') * Q(s', a')
        v_pi = np.sum(policy * q_table, axis=1)
        q_next = _bellman_q_sweep(transitions, n_states, n_actions, v_pi, gamma)
        delta = float(np.max(np.abs(q_next - q_table)))
        deltas.append(delta)
        q_table = q_next
        if delta < epsilon:
            converged = True
            break

    return ConvergenceResult(
        name=name,
        q_table=q_table,
        policy=policy.copy(),
        deltas=deltas,
        iterations=len(deltas),
        outer_iterations=1,
        converged=converged,
    )


def policy_iteration(
    env: gym.Env,
    initial_policy: np.ndarray,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
    max_outer_iterations: int = 200,
    max_eval_iterations: int = MAX_ITERATIONS,
    name: str = "policy_iteration",
) -> ConvergenceResult:
    """
    Ejecuta Policy Iteration alternando evaluación de Q^pi y mejora greedy.

    Parte de `initial_policy` y repite hasta que la política greedy no cambie.
    """
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    validate_policy(initial_policy, n_states, n_actions)

    policy = initial_policy.copy()
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)
    all_deltas: list[float] = []
    outer_iterations = 0
    converged = False

    for _ in range(max_outer_iterations):
        outer_iterations += 1
        eval_res = evaluate_policy_q(
            env,
            policy=policy,
            gamma=gamma,
            epsilon=epsilon,
            max_iterations=max_eval_iterations,
            initial_q=q_table,
            name=name,
        )
        q_table = eval_res.q_table
        all_deltas.extend(eval_res.deltas)

        improved_policy = greedy_policy_from_q(q_table)
        if np.array_equal(improved_policy, policy):
            converged = eval_res.converged
            break
        policy = improved_policy

    return ConvergenceResult(
        name=name,
        q_table=q_table,
        policy=policy,
        deltas=all_deltas,
        iterations=len(all_deltas),
        outer_iterations=outer_iterations,
        converged=converged,
    )


def q_value_iteration(
    env: gym.Env,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
    max_iterations: int = MAX_ITERATIONS,
    initial_q: np.ndarray | None = None,
    name: str = "q_value_iteration",
) -> ConvergenceResult:
    """
    Ejecuta Q-Value Iteration usando el operador de optimalidad de Bellman.

    Q_{k+1}(s, a) = sum_{s'} P(s'|s, a) * [r + gamma * (1 - done) * max_{a'} Q_k(s', a')]
    """
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    transitions = env.unwrapped.P

    q_table = (
        np.zeros((n_states, n_actions), dtype=np.float64)
        if initial_q is None
        else initial_q.astype(np.float64, copy=True)
    )
    deltas: list[float] = []
    converged = False

    for _ in range(max_iterations):
        v_star = np.max(q_table, axis=1)
        q_next = _bellman_q_sweep(transitions, n_states, n_actions, v_star, gamma)
        delta = float(np.max(np.abs(q_next - q_table)))
        deltas.append(delta)
        q_table = q_next
        if delta < epsilon:
            converged = True
            break

    return ConvergenceResult(
        name=name,
        q_table=q_table,
        policy=greedy_policy_from_q(q_table),
        deltas=deltas,
        iterations=len(deltas),
        outer_iterations=1,
        converged=converged,
    )


def evaluate_policy_episodes(
    env: gym.Env,
    policy: np.ndarray,
    n_episodes: int = TEST_EPISODES,
    seed: int = SEED,
) -> EvalMetrics:
    """
    Evalúa una política interactuando con Gymnasium mediante reset() y step().
    """
    n_actions = int(env.action_space.n)
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    steps_list: list[int] = []
    successes = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0

        while True:
            probs = policy[state]
            if np.max(probs) >= 1.0 - 1e-12:
                action = int(np.argmax(probs))
            else:
                action = int(rng.choice(n_actions, p=probs))

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            state = next_state

            if terminated or truncated:
                if terminated:
                    successes += 1
                break

        rewards.append(total_reward)
        steps_list.append(steps)

    return EvalMetrics(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_steps=float(np.mean(steps_list)),
        success_rate=float(successes / n_episodes),
    )


def _save_convergence_plot(
    results: list[ConvergenceResult],
    title: str,
    output_path: Path,
    epsilon: float = EPSILON,
) -> None:
    """Genera y guarda una gráfica en escala logarítmica de delta por iteración."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for res in results:
        iters = np.arange(1, len(res.deltas) + 1)
        ax.semilogy(iters, np.maximum(res.deltas, 1e-15), linewidth=1.8, label=res.name)

    ax.axhline(
        epsilon,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"Criterio ε = {epsilon:.0e}",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteración (barrido completo de Bellman)")
    ax.set_ylabel("Cambio máximo Δ = max |Q_{k+1} - Q_k| (escala log)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_comparison_plot(
    fixed_results: list[ConvergenceResult],
    output_path: Path,
) -> None:
    """Compara V(s) promedio y recompensa media en episodios por política."""
    names = [r.name for r in fixed_results]
    mean_v = [float(np.mean(np.sum(r.policy * r.q_table, axis=1))) for r in fixed_results]
    mean_q = [float(np.mean(r.q_table)) for r in fixed_results]
    mean_rewards = [
        r.eval_metrics.mean_reward if r.eval_metrics is not None else 0.0
        for r in fixed_results
    ]
    std_rewards = [
        r.eval_metrics.std_reward if r.eval_metrics is not None else 0.0
        for r in fixed_results
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax1.bar(x - width / 2, mean_q, width, label="Promedio global Q(s, a)", color="#4c72b0")
    ax1.bar(x + width / 2, mean_v, width, label="Promedio V^π(s)", color="#55a868")
    ax1.set_title("Valores Q y V estimados por política", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("Valor esperado")
    ax1.axhline(0.0, color="gray", linewidth=0.8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8.5)

    bars = ax2.bar(
        x,
        mean_rewards,
        yerr=std_rewards,
        capsize=4,
        color=["#c44e52", "#dd8452", "#8172b3", "#55a868"],
    )
    ax2.set_title("Recompensa promedio en 100 episodios (Taxi-v4)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("Recompensa acumulada por episodio")
    ax2.axhline(8.0, color="green", linestyle="--", linewidth=1.0, label="Umbral óptimo (~8.0)")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8.5)

    y_min = min(val - err for val, err in zip(mean_rewards, std_rewards)) - 95.0
    y_max = max(val + err for val, err in zip(mean_rewards, std_rewards)) + 85.0
    ax2.set_ylim(y_min, y_max)

    for bar, val, err in zip(bars, mean_rewards, std_rewards):
        if val >= 0:
            y_pos = val + err
            offset = 6
            va = "bottom"
        else:
            y_pos = val - err
            offset = -8
            va = "top"
        ax2.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _summarize_result(
    res: ConvergenceResult, q_star: np.ndarray
) -> dict[str, tt.Any]:
    """Construye un diccionario serializable con las estadísticas de un experimento."""
    v_pi = np.sum(res.policy * res.q_table, axis=1)
    return {
        "name": res.name,
        "iterations": res.iterations,
        "outer_iterations": res.outer_iterations,
        "converged": res.converged,
        "final_delta": res.deltas[-1] if res.deltas else None,
        "q_mean": float(np.mean(res.q_table)),
        "q_std": float(np.std(res.q_table)),
        "q_min": float(np.min(res.q_table)),
        "q_max": float(np.max(res.q_table)),
        "v_mean": float(np.mean(v_pi)),
        "max_diff_vs_q_star": float(np.max(np.abs(res.q_table - q_star))),
        "eval_metrics": asdict(res.eval_metrics) if res.eval_metrics else None,
    }


def _log_to_tensorboard(
    writer: SummaryWriter, tag_prefix: str, results: list[ConvergenceResult]
) -> None:
    """Registra curvas de convergencia delta y recompensas en TensorBoard."""
    for res in results:
        safe_name = res.name.replace(" ", "_").replace("(", "").replace(")", "")
        for step, delta in enumerate(res.deltas, start=1):
            writer.add_scalar(f"{tag_prefix}/delta/{safe_name}", delta, step)
        if res.eval_metrics is not None:
            writer.add_scalar(
                f"{tag_prefix}/mean_reward/{safe_name}",
                res.eval_metrics.mean_reward,
                res.iterations,
            )


def run_experiments() -> dict[str, tt.Any]:
    """Ejecuta todos los experimentos comparativos en Taxi-v4 y guarda artefactos."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    # Referencia óptima Q* mediante Q-Value Iteration partiendo de ceros
    qvi_zeros = q_value_iteration(env, name="Q0 = 0 (ceros)")
    q_star = qvi_zeros.q_table
    optimal_policy = qvi_zeros.policy

    # 1. Evaluación de Q^pi para diferentes políticas fijas
    fixed_policies = [
        ("Aleatoria uniforme", make_uniform_policy(n_states, n_actions)),
        ("Sesgada (movimiento)", make_biased_movement_policy(n_states, n_actions)),
        ("Epsilon-greedy (0.2)", make_epsilon_greedy_policy(q_star, epsilon=0.2)),
        ("Óptima (greedy)", optimal_policy),
    ]
    fixed_results: list[ConvergenceResult] = []
    for label, policy in fixed_policies:
        res = evaluate_policy_q(env, policy=policy, name=label)
        res.eval_metrics = evaluate_policy_episodes(eval_env, res.policy)
        fixed_results.append(res)

    # 2. Policy Iteration partiendo de distintas políticas iniciales pi_0
    initial_policies = [
        ("pi0: Aleatoria uniforme", make_uniform_policy(n_states, n_actions)),
        ("pi0: Siempre Sur (a=0)", make_fixed_action_policy(n_states, n_actions, 0)),
        ("pi0: Determinista aleatoria", make_random_deterministic_policy(n_states, n_actions)),
    ]
    pi_results: list[ConvergenceResult] = []
    for label, init_pi in initial_policies:
        res = policy_iteration(env, initial_policy=init_pi, name=label)
        res.eval_metrics = evaluate_policy_episodes(eval_env, res.policy)
        pi_results.append(res)

    # 3. Q-Value Iteration partiendo de distintas inicializaciones Q_0
    rng = np.random.default_rng(SEED)
    q_inits: list[tuple[str, np.ndarray | None]] = [
        ("Q0 = 0 (ceros)", None),
        ("Q0 = +20 (optimista)", np.full((n_states, n_actions), 20.0)),
        ("Q0 = -20 (pesimista)", np.full((n_states, n_actions), -20.0)),
        ("Q0 ~ U(-5, 5)", rng.uniform(-5.0, 5.0, size=(n_states, n_actions))),
    ]
    qvi_results: list[ConvergenceResult] = []
    for label, init_q in q_inits:
        res = qvi_zeros if init_q is None else q_value_iteration(env, initial_q=init_q, name=label)
        res.eval_metrics = evaluate_policy_episodes(eval_env, res.policy)
        qvi_results.append(res)

    # Gráficas PNG con Matplotlib
    _save_convergence_plot(
        fixed_results,
        "Convergencia de Q^π para políticas fijas en Taxi-v4",
        PLOTS_DIR / "convergencia_politicas_fijas.png",
    )
    _save_convergence_plot(
        pi_results,
        "Convergencia de Policy Iteration desde distintas políticas iniciales (π₀)",
        PLOTS_DIR / "convergencia_policy_iteration.png",
    )
    _save_convergence_plot(
        qvi_results,
        "Convergencia de Q-Value Iteration desde distintas inicializaciones (Q₀)",
        PLOTS_DIR / "convergencia_q_value_iteration.png",
    )
    _save_comparison_plot(
        fixed_results,
        PLOTS_DIR / "comparacion_q_y_recompensas.png",
    )

    # Registro en TensorBoard
    writer = SummaryWriter(log_dir=str(RUNS_DIR / "taxi_v4_analysis"))
    _log_to_tensorboard(writer, "fixed_policy_eval", fixed_results)
    _log_to_tensorboard(writer, "policy_iteration", pi_results)
    _log_to_tensorboard(writer, "q_value_iteration", qvi_results)
    writer.close()

    summary = {
        "env": ENV_NAME,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "test_episodes": TEST_EPISODES,
        "n_states": n_states,
        "n_actions": n_actions,
        "fixed_policy_evaluation": [
            _summarize_result(r, q_star) for r in fixed_results
        ],
        "policy_iteration": [_summarize_result(r, q_star) for r in pi_results],
        "q_value_iteration": [_summarize_result(r, q_star) for r in qvi_results],
    }

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    env.close()
    eval_env.close()
    return summary


def print_summary(summary: dict[str, tt.Any]) -> None:
    """Imprime un resumen tabular legible de los resultados en consola."""
    print(
        f"\n=== Resultados en {summary['env']} "
        f"(gamma={summary['gamma']}, epsilon={summary['epsilon']:.0e}, "
        f"episodios={summary['test_episodes']}) ==="
    )

    sections = [
        ("1. Evaluación de Q^pi (políticas fijas)", summary["fixed_policy_evaluation"]),
        ("2. Policy Iteration (distintas pi_0)", summary["policy_iteration"]),
        ("3. Q-Value Iteration (distintas Q_0)", summary["q_value_iteration"]),
    ]

    for title, rows in sections:
        print(f"\n{title}")
        print(
            f"{'Nombre':<28} {'Iter':>5} {'Ext':>4} {'Q_mean':>9} "
            f"{'V_mean':>9} {'Reward':>10} {'Éxito':>7} {'||Q-Q*||':>10}"
        )
        print("-" * 88)
        for row in rows:
            ev = row["eval_metrics"]
            print(
                f"{row['name']:<28} {row['iterations']:>5d} "
                f"{row['outer_iterations']:>4d} {row['q_mean']:>9.3f} "
                f"{row['v_mean']:>9.3f} {ev['mean_reward']:>10.2f} "
                f"{ev['success_rate'] * 100:>6.1f}% {row['max_diff_vs_q_star']:>10.2e}"
            )


if __name__ == "__main__":
    results_summary = run_experiments()
    print_summary(results_summary)
