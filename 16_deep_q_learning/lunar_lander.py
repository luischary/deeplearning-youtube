"""Treinamento e avaliação de agente Deep Q-Learning (DQN) no ambiente LunarLander-v3.

Estado (8 valores contínuos):
    [pos_x, pos_y, vel_x, vel_y, ângulo, vel_angular, perna_esq_chão, perna_dir_chão]

Ações (4 valores discretos):
    0 = não fazer nada
    1 = ligar motor lateral esquerdo
    2 = ligar motor principal (frear queda)
    3 = ligar motor lateral direito
"""

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from q_learning import QLearningAgentCompleto


def set_seed(seed: int = 50):
    """Fixa sementes aleatórias para facilitar a reprodutibilidade dos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(tamanho_estado: int, num_acoes: int) -> nn.Module:
    """Usa exatamente a mesma MLP do CartPole, adaptando apenas entrada (8) e saída (4)."""
    return nn.Sequential(
        nn.Linear(tamanho_estado, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_acoes),
    )


def train(
    agent,
    episodes: int = 1000,
    render: bool = False,
    initial_seed: int = 50,
):
    """Executa o loop principal de treinamento e validação a cada 100 episódios."""
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    train_metrics = []
    validation_metrics = []
    best_validation_score = float("-inf")

    for episode in range(episodes):
        # Seeds de treino a partir de 50 para separar dos episódios de validação (0..4)
        state, _ = env.reset(seed=initial_seed + episode)
        episode_over = False
        episode_reward = 0.0

        while not episode_over:
            # 1. Agente escolhe qual motor acionar (ou não fazer nada)
            action = agent.act(state)

            # 2. Ambiente avança um passo físico e retorna recompensa (combustível, pouso, queda)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_over = terminated or truncated

            # 3. Atualiza o agente usando a recompensa nativa do LunarLander
            agent.learn(state, action, reward, next_state, terminated)

            state = next_state

        train_metrics.append(float(episode_reward))

        # Avalia a cada 100 episódios completos (100, 200, ..., 1000) em modo guloso
        if (episode + 1) % 100 == 0:
            agent.eval()
            eval_score = evaluate(agent, render=False)
            agent.train()
            validation_metrics.append(eval_score)
            if eval_score > best_validation_score:
                best_validation_score = eval_score

        print(
            f"Episode {episode + 1:4d} | Train reward: {episode_reward:7.2f} | "
            f"Best val: {best_validation_score:7.2f} | Epsilon: {agent.epsilon:.4f}"
        )

    env.close()
    return train_metrics, validation_metrics, best_validation_score


def evaluate(agent, episodes: int = 5, render: bool = False) -> float:
    """Avalia o agente em 5 terrenos fixos de validação (seeds 0..4) e retorna a média."""
    scores = []
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)

    for episode in range(episodes):
        state, _ = env.reset(seed=episode)
        episode_over = False
        episode_reward = 0.0

        while not episode_over:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_over = terminated or truncated
            state = next_state

        scores.append(float(episode_reward))

    env.close()
    return float(np.mean(scores))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina um agente DQN no LunarLander-v3."
    )
    parser.add_argument(
        "--preset",
        choices=["padrao", "ajustado"],
        default="ajustado",
        help=(
            "'padrao' usa MSE, batch=64, memory=10k, target_update=100; "
            "'ajustado' (padrão) usa HuberLoss(delta=100), batch=256, memory=100k, target_update=250."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Quantidade de episódios de treinamento (padrão: 1000).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Não abre a janela gráfica na avaliação final.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(50)

    state_size = 8
    action_size = 4

    if args.preset == "padrao":
        # Mesmos hiperparâmetros do CartPole: útil para comparar o impacto da escala de rewards
        agent = QLearningAgentCompleto(
            build_func=build_model,
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=0.995,
            target_update_freq=100,
            batch_size=64,
            memory_size=10_000,
            loss_fn=nn.MSELoss(),
        )
    else:
        # Configuração ajustada para o LunarLander:
        # - HuberLoss(delta=100) evita gradientes explosivos quando ocorrem recompensas extremas
        # - Memória maior (100k), batch maior (256) e target mais estável (250) reduzem oscilações
        agent = QLearningAgentCompleto(
            build_func=build_model,
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=0.995,
            target_update_freq=250,
            batch_size=256,
            memory_size=100_000,
            loss_fn=nn.HuberLoss(delta=100.0),
        )

    train_metrics, validation_metrics, best_validation_score = train(
        agent,
        episodes=args.episodes,
        render=False,
    )

    # Desliga a exploração antes de assistir aos pousos finais do agente
    agent.eval()
    final_eval = evaluate(agent, render=not args.no_render)
    print(f"\nBest validation score: {best_validation_score:.2f}")
    print(f"Final evaluation score (5 episodes): {final_eval:.2f}")
    print(f"Validation history: {[round(v, 1) for v in validation_metrics]}")
