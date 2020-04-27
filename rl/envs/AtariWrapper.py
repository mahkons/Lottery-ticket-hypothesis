import torch
import torchvision
import torchvision.transforms as transforms
from collections import deque

from envs.EnvWrapper import EnvWrapper

class AtariWrapper(EnvWrapper):
    def __init__(self, name, random_state):
        super(AtariWrapper, self).__init__(name, random_state)
        self.state_sz = (4, 84, 84)

        # keeps 4 previous obseravations
        self.q = deque(maxlen=4)

    def transform_obs(self, obs):
        # Atari observation transformation from https://arxiv.org/pdf/1312.5602.pdf  
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((110, 84)),
            transforms.Lambda(lambda img: transforms.functional.crop(img, 18, 0, 84, 84)),
            transforms.ToTensor(),
        ])
        obs = transform(obs)
        self.q.append(obs)

        while len(self.q) < 4:
            self.q.append(obs)

        return torch.cat(list(self.q)).numpy()
