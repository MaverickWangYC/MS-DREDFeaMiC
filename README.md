Implementation of MS-DREDFeaMiC in the paper "An end-to-end mass spectrometry data classification model with a unified architecture"

Please refer to the supplementary materials of the paper for detailed parameters.

We conducted 10 comparative experiments on 7 public datasets and one self-constructed dataset.

Dataset 1-1: MI(pair mode)


python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=2 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='MI_pair' --K=3 --balance=0 --stop_flag='F_train'

Dataset 1-2: MI(joint mode)
python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=2 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='MI' --K=3 --balance=0 --stop_flag='F_train'

Dataset 2-1: CHD(pair mode)
python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=2 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='CHD_Paper_Pair' --K=4 --balance=0 --stop_flag='F_train'

Dataset 2-2: CHD(joint mode)
python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=2 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='CHD_Paper' --K=6 --balance=0 --stop_flag='F_train'

Dataset 3: tomato
python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=2 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='tomato' --K=1 --balance=0 --stop_flag='F_train'

Dataset 4: CC (self-constructed)
python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=2 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='CC_EW' --K=1 --balance=0 --stop_flag='F_train'

Dataset 5: KD
python train.py --epochs=XX --lr=XX --wd=XX --bs=XX --os=4 --ss=256 --ms=512 \
--label_smooth=XX --gamma=XX --loss='CEWithSmooth' --file='XX' --model='model_name' \
--seed=XX --dataset='KidneyDisease' --K=1 --balance=0 --stop_flag='F1_train'

Dataset 6: SIMS
python train_mouse.py --dataset='SIMS' --model='model_name'

Dataset 7: ICC_rms
python train_mouse.py --dataset='ICC_rms' --model='model_name'

Dataset 8: HIP_CER
python train_mouse.py --dataset='HIP_CER' --model='model_name'

