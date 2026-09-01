# NavRL 容器训练指南

## 路径映射

宿主机和容器使用同一份代码：

```text
宿主机：/home/yimingwei/NavRL
容器内：/workspace/NavRL
```

在宿主机修改代码后，容器内会立即看到变化。训练日志和检查点只要写入 `/workspace/NavRL`，也会保存在宿主机上。

## 首次创建容器

只有容器尚未创建时才需要执行。本配置向容器暴露全部 GPU，训练时再通过 `device=cuda:N` 选择具体显卡。

```bash
docker run -it \
  --name navrl-train \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 8g \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --mount type=bind,src=/home/yimingwei/NavRL,dst=/workspace/NavRL \
  -w /workspace/NavRL \
  --entrypoint /bin/bash \
  harbor.pudu.work/isaac_sim_2023/isaac_sim:2023.1.0
```

不要重复创建同名容器。可以先检查容器是否存在：

```bash
docker ps -a --filter name=navrl-train
```

## 进入容器

容器已停止时：

```bash
docker start -ai navrl-train
```

容器已在后台运行时：

```bash
docker exec -it navrl-train /bin/bash
```

进入后确认代码挂载正常：

```bash
cd /workspace/NavRL
git config --global --add safe.directory /workspace/NavRL
git status
```

## 配置 W&B

在线记录训练数据时，在容器中安全输入 W&B API key。不要把 key 写进仓库、命令或文档。

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo
export WANDB_API_KEY
```

确认变量已经设置，但不打印 key：

```bash
/isaac-sim/python.sh -c \
  'import os; print("WANDB_API_KEY is set" if os.getenv("WANDB_API_KEY") else "missing")'
```

如果不需要上传到 W&B，可以在训练命令中使用 `wandb.mode=offline`。

## 启动训练

进入训练目录：

```bash
cd /workspace/NavRL/isaac-training
```

使用 GPU 7，关闭评估和视频保存：

/isaac-sim/python.sh training/scripts/train.py device=cuda:7

## 关闭训练
docker exec navrl-train pkill -KILL -f 'training/scripts/train.py device=cuda:7'
docker exec navrl-train pkill -KILL -f 'training_delay/scripts/train.py device=cuda:7'
## 同时两个进程

export WANDB_API_KEY='你的新W&B密钥'

cd /workspace/NavRL/isaac-training
mkdir -p logs

nohup /isaac-sim/python.sh training/scripts/train.py \
  device=cuda:5 \
  enable_eval=false \
  record_eval_video=false \
  > logs/train_gpu6.log 2>&1 &

nohup /isaac-sim/python.sh training_delay/scripts/train.py \
  device=cuda:7 \
  enable_eval=false \
  record_eval_video=false \
  > logs/train_gpu7.log 2>&1 &
