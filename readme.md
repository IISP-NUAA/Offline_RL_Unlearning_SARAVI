# The code repository for  paper Offline Reinforcement Unlearning via Relative Advantage Function, NDSS27.

IISP@NUAA

## 1. Installation

Part of our codes is built upon the TrajDeleter repository, and it is recommended to go through the installation guideline in (https://github.com/2019ChenGong/TrajDeleter).  Please rename the `d3rlpy_ref` package in the repo to `d3rlpy` and move it to the `site_packages` folder in your conda environment (e.g., `/anaconda3/envs/<the-name-of-environment>/lib/python3.7/site-packages/`).

Our work are built upon the `Minari` interface, please run the following scripts in command line to download datasets (require installing Minari first).

    
    # pointmaze experiments
    minari download D4RL/pointmaze/large-dense-v2
    minari download D4RL/pointmaze/umaze-dense-v2
    minari download D4RL/pointmaze/medium-dense-v2

    # mujoco experiments
    minari download mujoco/halfcheetah/expert-v0
    minari download mujoco/halfcheetah/medium-v0
    minari download mujoco/hopper/expert-v0
    minari download mujoco/hopper/medium-v0
    minari download mujoco/walker2d/expert-v0

We will also upload the new Quad-X datasets to `Minari` later on, as well as checkpoints.

## 2.Obtain the fully trained model, the retrained model

    cd Offline_RL_processing

We use the `fully_training.py` script to produce $M_\mathrm{ori}$ and $M_\mathrm{ref}$, such as:

    # Fully training
    python fully_training.py --dataset pointmaze --n-steps 2000000 --algo IQL --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/iql_pointmaze_params.json --ratios 1.0 1.0 1.0 --seed 0

    # Retraining
    python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo IQL --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/iql_pointmaze_params.json --ratios 1.0 0.0 1.0 --seed 0

`--type Retrain` to distinguish runs. 

`--dataset`  choose from `[pointmaze, quadx, halfcheetah, hopper, walker2d]`

`--datasets` pick the datasets used in the task

``--algo`` choose from `[BCQ, BEAR, CQL, IQL, PLASP, TD3PLUSBC]`

``--n-steps`` specify steps used to build $M_\mathrm{ori}, M_\mathrm{ref}$

``--base-params`` specify param config file

``--ratios`` divide the forget set and retain set. The list of ratios should match the list of invloved datasets. For example, ``--ratios 0.9 1.0 1.0`` with ``--datasets large-dense-v2 umaze-dense-v2 medium-dense-v2`` means we randomly pick 10\% of ``large-dense-v2`` as $D_f$, the rest of ``large-dense-v2``, and the whole ``umaze-dense-v2`` and ``medium-dense-v2``, are set as $D_r$. It is recommended to set all `1`  when training the original model $M_\mathrm{ori}$, as we use random splitting for datasets with 0<ratio<1. 

We provide an illustration of training $M_\mathrm{ori}$ and $M_\mathrm{ref}$, as ``run_train_retrain_BCQ_on_pointmaze.sh`` in the folder. If the user found  model checkpoints occupy too much disk space, they can modify the save interval at the end of ``fully_training.py``, or using ``python clean_checkpoints.py --root [dir to clean]`` to delete intermediate checkpoints.

Changes to model structure or training configurations can be made by modifying the corresponding `*_params.json` file. For the PLAS-P algorithm, once the $M_\mathrm{ori}$ is learned, please change the `"warmup_steps"` configuration in `params.json`  of the trained model's dir to smaller positive integers or `0`, otherwise it will only update the model's conditional VAE during later unlearning transcations when the unlearning budget is less than `"warmup_steps"`. Please rename the final checkpoint in the `Fully_trained` folder to `model.pt`, for future unlearning.

## 3. Unlearning

    cd unlearning_processing
Use the following scripts to run unlearning methods on specified models.

### SARAVI
    python generate_SARAVI_script.py --algo [algo] --model-to-unlearn-dir [original model dir] --unlearning-steps [steps] --dataset [task] --datasets []

``--model-to-unlearn-dir`` the script generator will locate candidate folders within the ``--model-to-unlearn-dir`` that are built with ``--algo``. Passing dirs such as ``Offline_RL_processing/Fully_trained/pointmaze`` is fine.
We use the hardcoded ``RATIO_LIST`` in the head of ``generate_SARAVI_script.py``, which allows generating multiple unlearning tasks in the same time. Please modify them to run different tasks.

### Finetune
    python generate_Finetune_D4RL.py --algo [algo] --unlearning-steps [steps]  --dataset [task] --datasets []  --shuffle 1  --model-to-unlearn-dir [original model dir]
similar to SARAVI.

### TrajDeleter, Random Reward and Negative Reward

    python generate_unlearning_baselines_script.py --algo [algo] --total-steps [steps] --dataset [task]  --model-to-unlearn-dir [original model dir]
The command by default will generate scripts for random reward and TrajDeleter. The used datasets and ratio configurations can be modified by ``COMBINATIONS`` in the head of  ``generate_unlearning_baselines_script.py``. 
The codes are largely inherited from the TrajDeleter repo. We set 80\% of unlearning budgets to the first stage of TrajDeleter (``phase1_steps = int(total_steps * 0.8)``), and 20\% to its second stage. The user can set it to 100\% to switch to Negative Reward, and command the ``cmd_rewarding`` branch.

## 4. Evaluation

    cd Offline_RL_processing

#### Critic distance
    python generate_critic_divergence_measurement_script.py --model-dir-2 [dir of unlearned models]  --model-dir-1 [dir of retrained models] --dataset [task] --datasets [datasets]
The above script will generate critic evaluation scripts based on matched unlearned-retrain pairs (match is based on algorithm name and ratio). For instance, the user can set ``--model-dir-2`` as ``../unlearning_processing/SARAVI/pointmaze`` and set ``--model-dir-1`` as ``Retrain\pointmaze``. Additionally, the command can append configurations like ``--algo [algo like IQL]`` and ``--ratio [ratios like 0.9_1.0_1.0]`` to narrow the scope of evaluation. If the user wish to examine models unlearned with certain budgets, they can add ``--steps [e.g., 10000]`` to search model unlearned 10K steps. 

#### Policy distance
    python generate_policy_distance_measurement_script.py   --retrain-dir [dir of retrained models]  --unlearn-dir [dir of unlearned models]  --dataset [task] --datasets [datasets]
The command operates similarly with critic distance measurement.

#### ORLAuditor
    python generate_orl_auditor_script.py   --suspect-dir [dir of retrained or unlearned models] --shadow-training-steps [steps]   --shadow-dir [dir to store/search learned shadow policies]    --critic-search-dirs [dir to store/search learned critic]   --critic-save-dir [dir to store/search learned critic]    --dataset [task] --datasets [datasets]     --num-shadow-student 5  

#### Rollout 

    python generate_eval_script.py --root [dir-to-eval] --env-id-list [env-id]

``--root`` folder that contains models to evaluate, such as ``Retrain/pointmaze``
``--env-id-list`` describe dataset name in minari, then recover the environment of evaluation. It is set to three pointmaze environments by default.
The user can also use ``--algo``, ``--steps``, ``--ratio`` to narrow the evaluation scope. 

#### Ananlysis evaluaion results

    python analyze_policy_distance_results.py --root-dir stats_results_critic/stats_policy_distance_[task name]

    python analyze_critic_distance_results.py --root-dir stats_results_critic/stats_critic_[task name]

    python analyze_evaluation_results.py evaluation_results/[dataset prefix]_[task name]

    python analyze_orl_audit_results_NDSS_impl.py   --root-dir stats_results_orl_auditor/stats_orl_auditor_[task name]   --output-dir summary

Each script above generates mutliple tables: a **detail table** lists all identified results, a **ratio table** groups results on multiple seeds under the same ratio, and reports the mean values and statistics, a **summary table** reports summary of methods under mutiple seeds and settings.

#### Unlearning Efficiency

    python run_unlearning_cost_measurement_batch.py  --config unlearning_cost_batch_pointmaze.json  --parallel 1   --gpu-start 0

Users need to first train the TrajDeleter models for 10k steps (and the retrained models) to provide reference threshold, as shown in configurations  ``unlearning_cost_batch_pointmaze.json``.

    python report_unlearning_cost_including_failures.py   --batch-output ../unlearning_cost_batch_outputs/***

Use the command above to obtain statistics for unlearning efficiency of methods. Optimization steps are counted via the iterations that unlearning methods have consumed.