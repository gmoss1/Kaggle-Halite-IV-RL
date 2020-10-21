# Kaggle-Halite-IV-RL
Code to locally create and train an agent to play Halite IV, hosted by Kaggle - for rules of competition see link:
https://www.kaggle.com/c/halite

In summary the rules are 4 players on a board of size 21*21. Each player has some numbers of ships
(which can move, stay still, or become a shipyard), and shipyards, which can produce ships or skip their turn. 
All actions for all pieces are resolved simultaneously. The goal of the game is to collect as much of a resource,
'halite' as possible. Each square has some halite. Ships using their turn to stay still collect 25% of the halite 
on their respective square. Building ships and shipyards collect shipyards. Collisions between any two pieces lead to
all (or all but one) of them being destroyed, even if they are from the same player, with rules as specified in the link 
above. The game terminates after 400 moves, or earlier if only one player is still 'alive' (i.e has ships or shipyards
and enough halite to build a new ship).


The approach taken in this code is to train a neutral network to play this game. In particular, two neural networks are 
trained, such that one takes as input the observation of a single ship and recommends one of five actions (move in one of
the four directions North/East/South/West, stay still) and an analogous neural network for a shipyard recommending one of 
two actions (build ship/skip). 

This is a reinforcment learning problem. An environment is created which takes converts the game board information to an
array to use as input for the neural network, and then convert the resulting output into moves, and resolves these moves 
to produce the next set of "observations" for all relevant pieces. When a game is terminated for one of the pieces (i.e
it is destroyed or the game turn limit is reached), the environment also calculates a reward for each piece that played
in the episode so that the neural network could be updated. 
Training can be against random opponents or self-play.

The particular training algorithm used is Proximal Policy Optimization (PPO) - https://arxiv.org/abs/1707.06347

After training, the resulting neural network is saved (along with checkpoints throughout). A game is played (against
itself or basic hard-coded opponents) and then visualized.

As the training takes a long time to progress, a trained checkpoint is provided that can be used to view the results. Will need to specify
the path to the checkpoints in main.py to use them.
