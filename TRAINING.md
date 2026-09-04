# NavRL 容器训练指南

本文记录从 Harbor 镜像仓库下载 Isaac Sim 镜像、创建容器、挂载 NavRL 源码，以及启动训练的完整命令。

## 1. 拉取镜像

在宿主机执行。`docker pull` 会把镜像下载到本机 Docker 的镜像存储中，不会在当前目录生成一个同名文件夹。

```bash
docker pull harbor.pudu.work/isaac_sim_2023/isaac_sim:2023.1.0
```

确认镜像已经在本机：

```bash
docker images harbor.pudu.work/isaac_sim_2023/isaac_sim
```

应看到类似以下内容：

```text
REPOSITORY                                  TAG        IMAGE ID       SIZE
harbor.pudu.work/isaac_sim_2023/isaac_sim   2023.1.0  ...            19.9GB
```

## 2. 准备宿主机代码目录

本次使用的宿主机代码目录是 `/home/yimingwei/NavRL`。它必须在创建容器前存在，并且当前用户对其有读写权限：

```bash
mkdir -p /home/yimingwei/NavRL
ls -ld /home/yimingwei/NavRL
```

之前 `/mnt/nfs_b/weiyiming` 下创建目录没有权限，因此不要把没有写权限的目录作为挂载源。若以后获得了该目录的写权限，也可以把下面命令中的 `src` 改成对应路径。

## 3. 创建容器并挂载代码

只在容器第一次创建时执行。`--mount` 是 bind mount：宿主机的 `/home/yimingwei/NavRL` 和容器内的 `/workspace/NavRL` 指向同一份文件。

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

参数含义：

- `--gpus all`：让容器可以看到宿主机的全部 GPU，训练命令再用 `device=cuda:N` 选择具体 GPU。
- `--mount type=bind`：把宿主机代码目录挂载到容器。
- `-w /workspace/NavRL`：进入容器后默认工作目录。
- `--name navrl-train`：容器名称，后续用它执行 `docker start` 或 `docker exec`。

验证挂载是否生效：

```bash
# 宿主机执行
ls /home/yimingwei/NavRL/README.md

# 容器内执行
ls /workspace/NavRL/README.md
```

两边应当看到同一个文件。宿主机修改代码后，容器内会立即看到变化；容器内写入的日志和 checkpoint 也会出现在宿主机目录。

## 4. 再次进入容器

容器已停止时：

```bash
docker start -ai navrl-train
```

容器已在运行时：

```bash
docker exec -it navrl-train /bin/bash
```

检查容器和挂载：

```bash
docker ps -a --filter name=navrl-train
cd /workspace/NavRL
git config --global --add safe.directory /workspace/NavRL
git status
```

## 5. 配置 W&B

在线记录训练数据时，在容器内输入有效的 W&B API key。不要把 key 写进仓库、命令或文档：

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo
export WANDB_API_KEY
export WANDB_MODE=online
```

验证变量存在但不打印密钥：

```bash
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_API_KEY is set"; else echo "WANDB_API_KEY is missing"; fi
```

如果暂时不需要云端同步，可将训练命令中的 `wandb.mode=online` 改为 `wandb.mode=offline`。

## 6. 启动训练

进入训练目录：

```bash
cd /workspace/NavRL/isaac-training
mkdir -p logs
```

单进程示例（GPU7，关闭评估和视频保存）：

```bash
/isaac-sim/python.sh training/scripts/train.py \
  device=cuda:7 \
  enable_eval=false \
  record_eval_video=false \
  wandb.mode=online
```

同时启动两个独立进程：

```bash
cd /workspace/NavRL
WANDB_MODE=online ./start_training.sh
```

脚本会启动：

```text
training_delay -> GPU7 -> logs/train_delay_gpu7.log
training       -> GPU5 -> logs/train_gpu5.log
```

也可以手动启动：

```bash
cd /workspace/NavRL/isaac-training

nohup /isaac-sim/python.sh training_delay/scripts/train.py \
  device=cuda:7 \
  enable_eval=false \
  record_eval_video=false \
  wandb.mode=online \
  > logs/train_delay_gpu7.log 2>&1 &

nohup /isaac-sim/python.sh training/scripts/train.py \
  device=cuda:5 \
  enable_eval=false \
  record_eval_video=false \
  wandb.mode=online \
  > logs/train_gpu5.log 2>&1 &
```

查看进程、日志和 GPU：

```bash
docker exec navrl-train ps -eo pid,stat,etime,pcpu,pmem,cmd | \
  grep -E "training(_delay)?/scripts/train.py|training_delay/scripts/train.py"
tail -f /home/yimingwei/NavRL/isaac-training/logs/train_gpu5.log
tail -f /home/yimingwei/NavRL/isaac-training/logs/train_delay_gpu7.log
watch -n 2 nvidia-smi -i 5,7
```

## 7. 停止训练

只停止训练进程，保留容器：

```bash
docker exec navrl-train pkill -f 'training/scripts/train.py'
docker exec navrl-train pkill -f 'training_delay/scripts/train.py'
```

停止整个容器：

```bash
docker stop navrl-train
```

docker start navrl-train-gpu0-clean
docker exec -it navrl-train-gpu0-clean bash

cd /workspace/NavRL/isaac-training

export WANDB_MODE=offline
export NAVRL_ISAAC_EXPERIENCE=/isaac-sim/apps/omni.isaac.sim.python.gym.headless.kit
export NAVRL_ISAAC_ASSET_ROOT=/tmp/navrl-isaac-assets
export NAVRL_ISAAC_EXTENSIONS=omni.syntheticdata
export NAVRL_ISAAC_OFFLINE_ASSETS=1
export OMNI_DRONES_MINIMAL_IMPORTS=1
export OMNI_DRONES_DISABLE_VIEWPORT=1

/isaac-sim/python.sh training_delay/scripts/train.py \
  device=cuda:0 \
  headless=True \
  enable_eval=false \
  record_eval_video=false \
  wandb.mode=offline