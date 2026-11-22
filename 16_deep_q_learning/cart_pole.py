"""Treinamento e avaliação de agentes Deep Q-Learning no ambiente CartPole-v1.

Estado (4 valores contínuos):
    [posição_carrinho, velocidade_carrinho, ângulo_haste, velocidade_angular_haste]

Ações (2 valores discretos):
    0 = empurrar carrinho para a esquerda
    1 = empurrar carrinho para a direita
"""

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from q_learning import QLearningAgent, QLearningAgentCompleto


def set_seed(seed: int = 50):
    """Fixa sementes aleatórias para facilitar a reprodutibilidade dos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(tamanho_estado: int, num_acoes: int) -> nn.Module:
    """Cria uma MLP simples que mapeia o estado para um Q-value por ação.

    A camada de saída não usa ativação porque os Q-values são estimativas de
    retorno acumulado (números reais sem limite fixo).
    """
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
    use_reward_shaping: bool = True,
    initial_seed: int = 50,
):
    """Executa o loop principal de treinamento e validação periódica no CartPole-v1."""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    train_metrics = []
    validation_metrics = []
    best_validation_score = float("-inf")

    for episode in range(episodes):
        # Varia a seed a cada episódio (a partir de 50) para não coincidir com a validação (0..4)
        state, _ = env.reset(seed=initial_seed + episode)
        episode_over = False
        episode_reward = 0.0

        while not episode_over:
            # 1. Agente escolhe uma ação a partir do estado atual
            action = agent.act(state)

            # 2. Ambiente executa a ação e devolve o novo estado e a recompensa (+1 por passo)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            # terminated = perdeu (haste caiu / saiu dos limites); truncated = atingiu 500 passos
            episode_over = terminated or truncated

            # 3. Reward shaping (opcional): desconta a distância horizontal do centro (state[0]),
            # incentivando o carrinho a equilibrar a haste centralizado na tela
            learning_reward = reward - abs(float(state[0])) if use_reward_shaping else reward

            # Passamos `terminated` (e não `episode_over`) para que o agente só zere o futuro
            # quando realmente falha, e não quando o episódio apenas atinge o teto de 500 passos
            agent.learn(state, action, learning_reward, next_state, terminated)

            state = next_state

        train_metrics.append(float(episode_reward))

        # Avalia periodicamente com exploração desligada (modo eval)
        if episode % 50 == 0:
            agent.eval()
            eval_score = evaluate(agent)
            agent.train()
            validation_metrics.append(eval_score)
            if eval_score > best_validation_score:
                best_validation_score = eval_score

        print(
            f"Episode {episode:4d} | Train reward: {episode_reward:5.1f} | "
            f"Best val: {best_validation_score:5.1f} | Epsilon: {agent.epsilon:.4f}"
        )

    env.close()
    return train_metrics, validation_metrics, best_validation_score


def evaluate(agent, episodes: int = 5, render: bool = False) -> float:
    """Avalia o agente em episódios fixos de validação (seeds 0..episodes-1) e retorna a média."""
    scores = []
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

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
        description="Treina um agente Q-Learning / DQN no CartPole-v1."
    )
    parser.add_argument(
        "--agent",
        choices=["simples", "completo"],
        default="completo",
        help="Versão do agente: 'simples' (QLearningAgent) ou 'completo' (DQN com replay e target).",
    )
    parser.add_argument(
        "--sem-reward-shaping",
        action="store_true",
        help="Usa a recompensa pura do CartPole (+1/step) sem penalidade pela distância do centro.",
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

    state_size = 4
    action_size = 2

    if args.agent == "simples":
        agent = QLearningAgent(
            build_func=build_model,
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=0.995,
        )
    else:
        agent = QLearningAgentCompleto(
            build_func=build_model,
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=0.995,
            target_update_freq=100,
            loss_fn=nn.MSELoss(),
        )

    use_reward_shaping = not args.sem_reward_shaping
    train_metrics, validation_metrics, best_validation_score = train(
        agent,
        episodes=args.episodes,
        render=False,
        use_reward_shaping=use_reward_shaping,
    )

    # Desliga a exploração antes de assistir ao desempenho final do agente treinado
    agent.eval()
    final_eval = evaluate(agent, render=not args.no_render)
    print(f"\nBest validation score: {best_validation_score:.1f}")
    print(f"Final evaluation score (5 episodes): {final_eval:.1f}")
    print(f"Validation history: {[round(v, 1) for v in validation_metrics]}")
