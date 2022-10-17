GPUS=$1

# for cifar100 finetuning
# if measure memory, batch size = 128

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --nproc_per_node=$GPUS main.py  \
    --model deit_small_patch16_224  \
    --batch-size 128 \
    --data-path ../datasets/cifar100  \
    --data-set CIFAR \
    --input-size 224  \
    --exp_name debug \
    --num_workers 10 \
    --opt sgd \
    --lr 0.01 --unscale-lr \
    --weight-decay 0.0001 \
    --epoch 1000 \
    --ms_policy config/policy_tiny-8bit.txt
