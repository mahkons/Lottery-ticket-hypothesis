# Lottery ticket hypothesis
This repository contains an implementation of the article [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)  
And an application of this hypothesis to reinforcement learning
 - Supervised 
    - [x] Implement iterative magnitude pruning (IMP)  
    - [x] Test using toy net and dataset CIFAR10  
    - [ ] Test using VGG19 net and dataset CIFAR10  
    - [ ] Make it fast  
 - Reinforcement learning  
    - [x] Implement DQN  
    - [x] Test on classic gym environments (CartPole, LunarLander)   
    - [x] Try IMP (layerwise/global) with DQN on classic problems  
    - [ ] Analyze the specifics of applying Lottery ticket to DQN (e.g. target function updates)  
    - [ ] Try different early-stop techniques  
    - [ ] Dynamic epochs?  
    - [ ] DDPG? Dueling networks? Different RL architecures...  
    - [ ] Atari games?  
    - [ ] Compare with [Lottery tickets in RL and NLP](https://arxiv.org/abs/1906.02768) article?  

# Related articles
- Rewinding technique and stability analysis:  
[Stabilizing the Lottery Ticket Hypothesis](https://arxiv.org/abs/1903.01611), [Linear mode connectivity and the lottery ticket hypothesis](https://arxiv.org/abs/1912.05671)
- Application of hypothesis to reinforcement learning:  
[Lottery tickets in RL and NLP](https://arxiv.org/abs/1906.02768)
- Early-bird lottery tickets:  
[Drawing early-bird tickets](https://arxiv.org/abs/1909.11957)


### Something to read later
https://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon  
https://arxiv.org/abs/1506.02626  
https://arxiv.org/abs/1705.07565  
https://arxiv.org/abs/1711.02017  
https://arxiv.org/abs/1810.02054   
https://arxiv.org/abs/1810.05270  
https://arxiv.org/abs/1905.01067  
https://arxiv.org/abs/1907.04840  
https://arxiv.org/abs/2001.05050  
https://arxiv.org/abs/2002.00585  
https://arxiv.org/abs/2002.03875  

### Is it possible to make it fast?
https://arxiv.org/abs/1602.01528   
