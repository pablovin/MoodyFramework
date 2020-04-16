This repository holds the implementation of the Moody Framework for the Chef's Hat Card Game Simulation.

![Chef's Hat Card Game](gitImages/cardGame.jpg) 

## Moody Framework

The \emph Moody framework is able to explain the behavior of reinforcement learning agents in a competitive multiplayer card game scenario based on the players' assessment of its own performance.

It builds on the phenomenological confidence representation of the Q-values selection and implements a Growing-When-Required (GWR) network to establish a temporal impact between the taken actions. Also, the model allows each agent to measure their opponents' actions based on their own assessing, endowing them with a closed-world representation of the entire game.

#### Chef's Hat Card Game Simulation

Chef's Hat is a competitive card game  designed with specific HRI requirements in mind, which allows it to be followed and modeled by artificial agents with ease. The game mechanics were designed to evoke different social behavior between the players. 
Fora a complete overview on the development of the game, refer to:

- The Chef's Hat Simulation Environment for Reinforcement-Learning-Based Agents (https://arxiv.org/abs/2003.05861)
- The Chef's Hat Simulation Environment Repository: https://github.com/pablovin/ChefsHatGYM


## The Moody Framework Plugin


The Moody Framework implementation is made to be used as a Plugin to the Chef's Hat Simulation environment.
Every agent implemented by the simulation environment can be enhanced with a Moody plugin, allowing it to explain
its own behavior.

Each plugin is able to generate a mood network for each agent and its opponents. You can define this
at the agent instantiation phase:

```
intrinsicWithMoodDQL = Intrinsic(selfConfidenceType=CONFIDENCE_PHENOMENOLOGICAL, isUsingSelfMood=True,isUsingOponentMood=True)
```

## Use and distribution policy

All the examples in this repository are distributed under a Non-Comercial license. If you use this environment, you have to agree with the following itens:

- To cite our associated references in any of your publication that make any use of these examples.
- To use the environment for research purpose only.
- To not provide the environment to any second parties.

## Citations

- Barros, P., Sciutti, A., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2020). It's Food Fight! Introducing the Chef's Hat Card Game for Affective-Aware HRI. Accepted at the HRI2020
  Workshop on Exploring Creative Content in Social Robotics! arXiv preprint arXiv:2002.11458.

- Barros, P., Sciutti, A., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2020) The Chef's Hat Simulation Environment for Reinforcement-Learning-Based Agents. arXiv preprint arXiv:2003.05861.

- Barros, P., Tanevska, A., & Sciutti, A. (2020). Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game. arXiv preprint arXiv:2004.04000.
## Contact

Pablo Barros - pablo.alvesdebarros@iit.it

- [http://pablobarros.net](http://pablobarros.net)
- [Twitter](https://twitter.com/PBarros_br)
- [Google Scholar](https://scholar.google.com/citations?user=LU9tpkMAAAAJ)
