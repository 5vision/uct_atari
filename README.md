# UCT search for atari games

This repo contains code for running uct search on atari games, collecting uct search data in form (frame, best_uct_action) and then use it in supervised training. 

## UCT search

To run just uct search use:
  
    $ python run_uct.py

To run uct search and collect data for further training use:

    $ python collect_uct_data.py
 
## Supervised learning

To run supervised training with collected data use:

    $ python train_keras.py dirs_with_uct_data

Use flag --help to see all options for UCT and supervised learning scripts.