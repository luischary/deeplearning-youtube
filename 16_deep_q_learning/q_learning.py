"""Implementação didática do Deep Q-Learning (DQN) do zero em PyTorch.

Contém duas versões progressivas apresentadas no vídeo:
1. QLearningAgent: versão básica (rede única, atualização passo a passo).
2. QLearningAgentCompleto: DQN com Experience Replay e Target Network.
"""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn


class QLearningAgent:
    """Agente Q-Learning básico com aproximação por rede neural.

    Usa apenas uma rede neural e atualiza os pesos imediatamente a cada
    transição (s, a, r, s', done), sem replay buffer nem target network.
    """

    def __init__(
        self,
        build_func,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_init: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        # Hiperparâmetros do Q-Learning
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma                # Fator de desconto para recompensas futuras
        self.epsilon = epsilon_init       # Probabilidade inicial de explorar (ação aleatória)
        self.epsilon_decay = epsilon_decay  # Decaimento de epsilon ao fim de cada episódio
        self.epsilon_min = epsilon_min    # Limite mínimo de exploração
        self.mode = "train"

        # Rede neural que estima Q(s, a) para todas as ações possíveis
        self.model = build_func(state_size, action_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.loss = nn.MSELoss()

    def act(self, state) -> int:
        """Escolhe uma ação usando a política epsilon-greedy."""
        # Exploração: durante o treino, sorteia uma ação aleatória com probabilidade epsilon
        if self.mode == "train" and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))

        # Explotação: escolhe a ação com maior Q-value estimado pela rede
        # unsqueeze(0) adiciona a dimensão de batch: [state_size] -> [1, state_size]
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def train(self):
        """Ativa modo de treino (habilita exploração epsilon-greedy)."""
        self.mode = "train"
        self.model.train()

    def eval(self):
        """Ativa modo de avaliação (desliga exploração; política 100% gulosa)."""
        self.mode = "eval"
        self.model.eval()

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        """Atualiza a rede a partir de uma única transição (Temporal Difference)."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32)

        # Alvo de Bellman (sem gradientes): y = r + gamma * max_a' Q(s', a') * (1 - done)
        # Se done=True, (1 - done) vira 0 e zera a expectativa de recompensa futura.
        with torch.no_grad():
            next_max_q = torch.max(self.model(next_state_tensor), dim=1).values
            target = reward_tensor + self.gamma * next_max_q * (1.0 - done_tensor)

        # Estimativa atual Q(s, a) apenas para a ação que foi de fato executada
        q_values = self.model(state_tensor)
        current_q_value = q_values[0, action].unsqueeze(0)

        # Passo padrão de treinamento em PyTorch
        loss = self.loss(current_q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduz epsilon apenas ao final do episódio (e não a cada passo)
        if self.mode == "train" and self.epsilon > self.epsilon_min and done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgentCompleto:
    """Agente Deep Q-Network (DQN) com Experience Replay e Target Network.

    Resolve duas fontes clássicas de instabilidade do Q-Learning com redes neurais:
    - Experience Replay: guarda transições e treina em mini-batches aleatórios,
      quebrando a correlação temporal entre passos consecutivos.
    - Target Network: mantém uma cópia congelada dos pesos para calcular o alvo TD,
      evitando que o alvo se mova a cada atualização da rede online.
    """

    def __init__(
        self,
        build_func,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_init: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10_000,
        target_update_freq: int = 100,
        batch_size: int = 64,
        loss_fn: nn.Module | None = None,
    ):
        # Hiperparâmetros do Q-Learning
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.mode = "train"

        # Rede online (treinada a cada passo) e Target Network (cópia sincronizada periodicamente)
        self.model = build_func(state_size, action_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.target = build_func(state_size, action_size)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        # Permite alternar a função de perda (ex.: MSELoss no CartPole, HuberLoss no LunarLander)
        self.loss = loss_fn if loss_fn is not None else nn.MSELoss()

        # Controle da frequência de sincronização da Target Network
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Replay Buffer (FIFO): quando atinge memory_size, descarta automaticamente a transição mais antiga
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size

    def act(self, state) -> int:
        """Escolhe uma ação usando a rede online com política epsilon-greedy."""
        if self.mode == "train" and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def train(self):
        """Ativa modo de treino (habilita exploração epsilon-greedy)."""
        self.mode = "train"
        self.model.train()

    def eval(self):
        """Ativa modo de avaliação (desliga exploração; política 100% gulosa)."""
        self.mode = "eval"
        self.model.eval()

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        """Armazena a transição na memória e atualiza a rede usando um mini-batch aleatório."""
        # 1. Guarda a transição (s, a, r, s', done) no replay buffer
        self.memory.append((state, action, reward, next_state, done))

        # 2. Só começa a treinar quando houver exemplos suficientes para um batch completo
        if len(self.memory) < self.batch_size:
            return

        # 3. Amostra um mini-batch aleatório e separa cada campo em sua própria sequência
        # batch é uma lista de tuplas [(s1, a1, r1, s1', d1), ...]; zip(*batch) transpõe isso
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32)
        next_states_tensor = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32)

        # 4. Calcula o alvo TD em batch usando a Target Network congelada
        with torch.no_grad():
            next_max_q = torch.max(self.target(next_states_tensor), dim=1).values
            target = rewards_tensor + self.gamma * next_max_q * (1.0 - dones_tensor)

        # 5. Calcula Q(s, a) na rede online para o batch inteiro: shape [B, action_size]
        q_values = self.model(states_tensor)
        # gather seleciona, em cada linha do batch, o Q-value da ação realmente tomada:
        # actions_tensor [B] -> unsqueeze(1) [B, 1] -> gather -> [B, 1] -> squeeze(1) [B]
        current_q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # 6. Calcula a loss média no batch e atualiza os pesos da rede online
        loss = self.loss(current_q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1

        # 7. A cada target_update_freq passos, copia os pesos da rede online para a Target Network
        if self.mode == "train" and self.update_counter % self.target_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

        # 8. Reduz epsilon ao fim do episódio para explorar menos conforme o agente aprende
        if self.mode == "train" and self.epsilon > self.epsilon_min and done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
