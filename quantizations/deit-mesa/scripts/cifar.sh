EXP=$1

# for cifar100 finetuning
# if measure memory, batch size = 128

CUDA_VISIBLE_DEVICES=5,6 torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nproc_per_node=2 main.py  \
    --model deit_small_patch16_224  \
    --batch-size 384 \
    --data-path ../datasets/cifar100  \
    --data-set CIFAR \
    --input-size 224  \
    --exp_name deit_small_patch16_224_autocast_mesa \
    --num_workers 10 \
    --opt sgd \
    --lr 0.01 --unscale-lr \
    --weight-decay 0.0001 \
    --epoch 1000 \
    --ms_policy config/policy_tiny-8bit.txt
    --finetune deit_small_patch16_224-cd65a155.pth