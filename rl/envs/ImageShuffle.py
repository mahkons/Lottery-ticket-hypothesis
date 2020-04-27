import numpy as np

class ImageShuffle():
    def __init__(self, pieces, permutation, env, random_state):
        self.env = env(random_state=random_state)
        self.permutation = np.array(permutation) if permutation is not None else None
        self.pieces = pieces

        assert(self.env.state_sz[0] % self.pieces == 0)
        assert(self.env.state_sz[1] % self.pieces == 0)

        if self.permutation is None:
            self.permutation = np.random.permutation(self.pieces ** 2).reshape((self.pieces, self.pieces))

        self.state_sz = self.env.state_sz
        self.action_sz = self.env.action_sz
    
    def shuffle_image(self, obs):
        out_obs = list()
        for i in range(obs.shape[0]):
            image_blocks = np.concatenate(list(map(lambda x: np.split(x, self.pieces, 1), np.split(obs[i], self.pieces, 0)))) 
            image_blocks = image_blocks[self.permutation]

            h, w = obs.shape[1] // self.pieces, obs.shape[2] // self.pieces
            image_blocks = np.array([[image_blocks[self.pieces * (i // h) + j // w][i % h][j % w] \
                    for j in range(obs.shape[2])] \
                    for i in range(obs.shape[1])])

            out_obs.append(image_blocks)
        return np.stack(out_obs)

    # All remaining methods -- just delegation

    def step(self, action):
        obs, reward, done, info, real_reward = self.env.step(action)
        obs = self.shuffle_image(obs)
        return obs, reward, done, info, real_reward

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def get_steps(self):
        return self.env.get_steps()

    def get_total_reward(self):
        return self.env.get_total_reward()
