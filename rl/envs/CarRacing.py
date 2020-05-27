from envs.EnvWrapper import EnvWrapper

class CarRacing(EnvWrapper):
    def __init__(self, random_state):
        super(CarRacing, self).__init__("CarRacing-v0", random_state)
        self.state_sz = (64, 56)

    def transform_obs(self, obs):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: transforms.functional.crop(img, 12, 0, 66, 64)),
            transforms.Resize((64, 56)),
            transforms.ToTensor(),
        ])
        return transform(obs)

    # TODO
    # make descrete
    def step():
        pass

