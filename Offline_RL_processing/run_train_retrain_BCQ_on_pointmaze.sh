set -e
set -o xtrace
echo '20 tasks'

python fully_training.py --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 1.0 1.0 --seed 0
     
python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 0.0 1.0 --seed 0

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.9 1.0 1.0 --seed 0

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.0 1.0 1.0 --seed 0
     
python fully_training.py --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 1.0 1.0 --seed 1024
     
python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 0.0 1.0 --seed 1024

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.9 1.0 1.0 --seed 1024

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.0 1.0 1.0 --seed 1024

python fully_training.py --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 1.0 1.0 --seed 42
     
python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 0.0 1.0 --seed 42

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.9 1.0 1.0 --seed 42

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.0 1.0 1.0 --seed 42


python fully_training.py --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 1.0 1.0 --seed 46
     
python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 0.0 1.0 --seed 46

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.9 1.0 1.0 --seed 46

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.0 1.0 1.0 --seed 46

python fully_training.py --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 1.0 1.0 --seed 47
     
python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 1.0 0.0 1.0 --seed 47

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.9 1.0 1.0 --seed 47

python fully_training.py --type Retrain --dataset pointmaze --n-steps 2000000 --algo BCQ --datasets large-dense-v2 umaze-dense-v2 medium-dense-v2 --base-params ./params/bcq_pointmaze_params.json --ratios 0.0 1.0 1.0 --seed 47
     
