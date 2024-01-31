import numpy as np
import torch
from torch import nn


class SumChars(nn.Module):
    def __init__(self, obs_size, action_size, word_list, word_max_len, n_hidden: int = 1, hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        force_word_max_len = word_max_len
        force_word_list = [x for x in word_list if len(x)<=force_word_max_len]
        print(len(force_word_list))
        words_width = force_word_max_len*26

        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, words_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)
        word_array = np.zeros((words_width, len(force_word_list)))
        for i, word in enumerate(force_word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1 # word_array[j,i] = ord(c) # 
        self.words = torch.Tensor(word_array)

        self.action_last = nn.Sequential(
            nn.Linear(len(force_word_list)+obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,action_size)
        )

    def forward(self, x):
        y = self.f0(x.float())
        a = torch.log_softmax(torch.tensordot(y,self.words.to(self.get_device(y)), dims=((1,), (0,))),dim=-1)
        return self.action_last(torch.concat((a,x.float()),dim=1))

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index