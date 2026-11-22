"""Script introdutório para testar a API básica do Gymnasium no ambiente CartPole-v1.

Aqui ainda não há nenhum agente inteligente: o carrinho apenas escolhe ações
aleatórias para demonstrar o ciclo padrão reset() -> step() -> done.
"""

import time

import gymnasium as gym


def main():
    # Cria o ambiente CartPole com render_mode="human" para exibir a janela gráfica
    env = gym.make("CartPole-v1", render_mode="human")

    # Reseta o ambiente e recebe a primeira observação (estado inicial)
    # observation: [posição_carrinho, velocidade_carrinho, ângulo_haste, velocidade_angular]
    observation, info = env.reset()
    print(f"Observação inicial: {observation}")

    episode_over = False
    total_reward = 0.0

    while not episode_over:
        # Escolhe uma ação aleatória entre as possíveis (0 = esquerda, 1 = direita)
        action = env.action_space.sample()

        # Envia a ação ao ambiente e recebe a resposta desse passo:
        # - observation: novo estado após a ação
        # - reward: +1 por cada passo com a haste equilibrada
        # - terminated: True se a haste caiu ou o carrinho saiu dos limites (derrota)
        # - truncated: True se atingiu o limite máximo de 500 passos do episódio
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

        # Pequena pausa apenas para conseguirmos enxergar os passos na tela
        time.sleep(0.1)

    print(f"Episódio encerrado! Recompensa total: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
