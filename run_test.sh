
# eval refcoco setting
MODEL_PATH="magnet_refcoco.pth"
IMG_SIZE=480
python test.py --dataset="refcoco" --split="val" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco" --split="testA" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco" --split="testB" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}

# eval refcoco+ setting
MODEL_PATH="magnet_refcoco+.pth"
IMG_SIZE=480
python test.py --dataset="refcoco+" --split="val" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco+" --split="testA" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco+" --split="testB" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}

# eval refcocog umd setting
MODEL_PATH="magnet_refcocog_umd.pth"
IMG_SIZE=480
python test.py --dataset="refcocog" --split="val" --splitBy="umd" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcocog" --split="test" --splitBy="umd" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}

# eval refcocog google setting
MODEL_PATH="magnet_refcocog_google.pth"
IMG_SIZE=480
python test.py --dataset="refcocog" --split="val" --splitBy="google" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}

# eval extra/multiple datasets setting
MODEL_PATH="magnet_all.pth"
IMG_SIZE=448 # slightly smaller image size to speed up training
python test.py --dataset="refcoco" --split="val" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco" --split="testA" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco" --split="testB" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco+" --split="val" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco+" --split="testA" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcoco+" --split="testB" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcocog" --split="val" --splitBy="umd" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}
python test.py --dataset="refcocog" --split="test" --splitBy="umd" --resume=${MODEL_PATH} --workers="4" --ddp_trained_weights --window12 --img_size=${IMG_SIZE}

