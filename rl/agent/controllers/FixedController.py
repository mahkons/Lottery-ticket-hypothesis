class FixedController:
    def __init__(self, oracle):
        self.oracle = oracle

    def select_action(self, state, explore):
        return self.oracle(state, explore)

    def optimize(self):
        pass

    def optimization_completed(self):
        return True

    def prune(self):
        pass

    def reinit(self):
        pass

    def push_in_memory(self, state, action, next_state, reward, done):
        pass

    def get_state(self):
        return None
