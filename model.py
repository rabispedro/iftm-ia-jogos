import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# segunda parte


class Linear_QNet(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		# inicializar a rede neural
		super().__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(input_size, hidden_size)


	def forward(self, x):
		"""
		x é o tensor de entrada
		pytorch precisa de uma forward function
		vamos passar o tensor de entrada pela linear layer
		e tambem precisamos de uma Activation function (relu)
		"""
		x = F.relu(self.linear1(x))

		x = self.linear2

		return x

	def save(self, file_name="model.pth"):
		# salvar o modelo

		model_folder_path = "./model"
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)


class QTrainer:
	def __init__(self, model, lr, gamma):
		"""inicializa o QTrainer"""

		self.lr = lr
		self.gamma = gamma
		self.model = model

		self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss

	def train_step(self, state, action, reward, next_state, done):
		"""Treina o modelo com o jogo atual"""

		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)

		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)

		if len(state.shape) == 1:
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			done = (done, )

		# 1: valores Q previstos com o estado atual
		pred = self.model(state)

		target = pred.clone()
		# itera sobre o tensor
		for idx in range(len(done)):
			Q_new = reward[idx]
			if not done[idx]:
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

			target[idx][torch.argmax(action[idx]).item()] = Q_new

		# Markov Decision Process -> no lugar de bellman puro
		# 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
		# pred.clone()
		# preds[argmax(action)] = Q_new
		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward()

		self.optimizer.step()