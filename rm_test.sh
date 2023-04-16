

python expose_model.py --config-file configs/diffinst.coco.res101.yaml --eval-only MODEL.WEIGHTS models/torchvision-R-101.pkl
#training 
# python train_net.py --num-gpus 1 \
#     --config-file configs/diffinst.coco.res101.yaml \
#     MODEL.WEIGHTS models/torchvision-R-101.pkl

#testing 
# python train_net.py --num-gpus 1 \
    # --config-file configs/diffinst.coco.res101.yaml \
    # --eval-only MODEL.WEIGHTS models/torchvision-R-101.pkl