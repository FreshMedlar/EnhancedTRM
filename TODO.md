
reinforcement learning
- [x] train a teacher model
- [ ] train a teacher model that uses vision (this is difficult)
- [ ] create a base dreamer model
- [x] use trm to learn the world model (action conditioning works better)
- [ ] create a model based rl algo that uses the trm world model

- [ ] create hierarchical representation of the world model to have high level goals

# close next
- [ ] rnn to select vision patches
        problem: the world model can't predict the part he can't even see because it's not even passed by the rnn
- [ ] attn block to analyze patches 
