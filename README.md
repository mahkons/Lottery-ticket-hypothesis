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
    - [x] IMP with rewind
    - [x] Early stop criterions
    - [x] Rescaling weight after reinit
    - [ ] Global/Layerwise/ERK
    - [ ] Analyze the specifics of applying Lottery ticket to DQN (e.g. target function updates)  
    - [ ] Try different early-stop techniques  
    - [ ] Dynamic epochs?  
    - [ ] DDPG? Dueling networks? Different RL architecures...  
    - [ ] Atari games?  
    - [ ] Compare with other articles?
    - [ ] Clean up this list  

# Related articles
- Rewinding technique and stability analysis:  
[Stabilizing the Lottery Ticket Hypothesis](https://arxiv.org/abs/1903.01611), [Linear mode connectivity and the lottery ticket hypothesis](https://arxiv.org/abs/1912.05671)  
- Application of hypothesis to reinforcement learning:  
[Lottery tickets in RL and NLP](https://arxiv.org/abs/1906.02768)  
- Early-bird lottery tickets:  
[Drawing early-bird tickets](https://arxiv.org/abs/1909.11957)  
- Pruning in RL:  
[Policy Pruning and Shrinking for Deep Reinforcement Learning](https://arxiv.org/abs/2001.05012)  


### More or less related
# TODO Mark interesting ones somehow
[Optimal Brain Surgeon](https://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon)--second derivatives  
[Learning both Weights and Connections](https://arxiv.org/abs/1506.02626)--prune + tune  
[Dynamic Network Surgery](https://arxiv.org/abs/1608.04493)--parameter importance + grow pruned?  
[Layerwise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565)--layerwise second derivatives  
[Grow and Prune Tool](https://arxiv.org/abs/1711.02017)--??  
[Adaptive sparse connectivity](https://www.nature.com/articles/s41467-018-04316-3) -- ?? TODO  
[Overparametrized networks provably optimized](https://arxiv.org/abs/1810.02054)--gradient descent on overparametrized networks  
[Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)--structured with random reinit  
[Deconstructing Lottery Tickets](https://arxiv.org/abs/1905.01067)--lottery ticket signs + supermasks  
[Sparse Networks from Scratch](https://arxiv.org/abs/1907.04840)--sparse momentum  
[Making All Tickets Winners](https://arxiv.org/pdf/1911.11134.pdf) -- ?? RIGL  
[On Iterative Neural Network Pruning](https://arxiv.org/abs/2001.05050)--pruning methods summary  
[Proving the Lottery Ticket](https://arxiv.org/abs/2002.00585)--??  
[Improving Reliability of Lottery Tickets](https://arxiv.org/abs/2002.03875)--??  

### Is it possible to make it fast?
https://arxiv.org/abs/1602.01528   
