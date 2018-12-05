# ActorMimic baseline
We use [ActorMimic](https://github.com/eparisotto/ActorMimic) as a baseline for transfer learning experiments.

## Training a distilled network using ActorMimic
The script `./run.sh` is used to train a single network that distills information from the experts for multiple games. (You can change this list in `run.sh` and `scripts/run_amn_polreg_paper`. NOTE: Make the change in both files, the first file launches the java environments for the games entered, the second trains the actormimic network for the games entered)

To generate the distilled network, run ``./run.sh $START_PORT $exp_folder`` The model will be stored as `NeuralActorMimicLearner_polreg_paper_$steps.t7` every 250,000 steps. 

To see how the distilled network performs on each game, you can see the avg_rewards vs timestep logs in `logs\$exp_folder\$game_id\test_avgR.log.eps` where `$game_id` is the index of the game in the list of games you entered.

## Evaluating for transfer
The script `./run_transfer.sh` can be used to evaluate the ActorMimic baseline on the target game(s). 

To evaluate, run ``./run_transfer.sh $START_PORT $exp_folder $transfer_type $seed`` 
Here `$transfer_type` = 0 if we initialise using the distilled network, 1 if we use a standard initialisation of a single-task network, 2 if we use standard initialisation of a multi-task network. This generates avg_rewards vs timestep logs for each network, which are stored in `logs\$exp_folder_transfer_seed_$seed_t_$transfer_type\1\test_avgR.log` (transfer_type = 0 produces the best results).

	
