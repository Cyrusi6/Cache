# Kubernetes GPU 任务使用指南

本工具把训练和评测封装为 Kubernetes Job。只需声明 GPU 数量，Kubernetes 会分配空闲显卡；资源不足时任务保持 `Pending`，待显卡释放后自动运行。

## 默认配置

- context：`default`
- namespace：`c2c-research`
- 节点：`4090-24gx4`
- 镜像：`swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- 资源：每张 GPU 默认申请 8 CPU、32Gi 内存
- 最长运行时间：72 小时
- manifest：`local/k8s/manifests/<完整任务名>.json`

任务名会自动追加时间戳。提交后保存输出中的 `job=...`，后续查看日志、等待和删除均使用这个完整名称。

当前实现挂载本机 `hostPath`，因此 `--node` 只接受 `4090-24gx4`，不能把同一任务安全投递到其他服务器。

默认 PyTorch 镜像使用 Python 3.11、PyTorch 2.6.0+cu124；它满足项目的 Python ≥3.10 要求，但不等同于宿主机 `c2c-py310-cu124`。必须严格使用 Python 3.10 时，请通过 `--image` 提供固定版本的自定义镜像。

## 初始化

首次使用执行：

```bash
bash bash/k8s/gpu_job.sh init
```

该命令创建 `c2c-research` namespace 和本地缓存目录，并检查节点状态、GPU 数量与 Job 权限。

## 提交任务

建议先做服务端检查，不会创建 Job：

```bash
bash bash/k8s/gpu_job.sh submit \
  --name smoke \
  --gpus 1 \
  --dry-run \
  -- python -c 'import torch; print(torch.cuda.is_available())'
```

单卡训练：

```bash
bash bash/k8s/gpu_job.sh submit \
  --name v23-seed0 \
  --gpus 1 \
  --follow \
  -- python script/train/SFT_train.py --config <train-config>
```

四卡联合训练：

```bash
bash bash/k8s/gpu_job.sh submit \
  --name v23-four-gpu \
  --gpus 4 \
  -- torchrun --nproc_per_node=4 \
     script/train/SFT_train.py --config <train-config>
```

四个单卡实验可连续提交，Kubernetes 会并行调度到四张卡：

```bash
bash bash/k8s/gpu_job.sh submit --name exp-a --gpus 1 -- python script/train/SFT_train.py --config <config-a>
bash bash/k8s/gpu_job.sh submit --name exp-b --gpus 1 -- python script/train/SFT_train.py --config <config-b>
bash bash/k8s/gpu_job.sh submit --name exp-c --gpus 1 -- python script/train/SFT_train.py --config <config-c>
bash bash/k8s/gpu_job.sh submit --name exp-d --gpus 1 -- python script/train/SFT_train.py --config <config-d>
```

评测示例：

```bash
bash bash/k8s/gpu_job.sh submit \
  --name eval-v23 \
  --gpus 1 \
  -- python script/evaluation/unified_evaluator.py --config <eval-config>
```

可用 `--cpu 16`、`--memory 64Gi` 覆盖默认资源，用 `--timeout-hours 96` 修改 Job 截止时间。

## 管理任务

```bash
# 查看本工具创建的 Job 和 Pod
bash bash/k8s/gpu_job.sh list

# 查看日志；--follow 持续跟随
bash bash/k8s/gpu_job.sh logs <完整任务名> --follow

# 查看 Job 状态和事件
bash bash/k8s/gpu_job.sh describe <完整任务名>

# 等待完成
bash bash/k8s/gpu_job.sh wait <完整任务名> --timeout 72h

# 删除任务及其 Pod；只允许删除本工具管理的 Job
bash bash/k8s/gpu_job.sh delete <完整任务名>
```

安全删除还要求对应文件存在于 `local/k8s/manifests/`。不要在任务运行期间手工删除该 manifest。

使用其他 namespace 时，初始化和每次操作都显式传入：

```bash
bash bash/k8s/gpu_job.sh init --namespace <namespace>
bash bash/k8s/gpu_job.sh list --namespace <namespace>
```

不要使用其他项目的 namespace。

## 首次 bootstrap 与缓存

默认镜像启动后会在 `local/k8s/runtime/envs/<依赖指纹>/venv` 创建不可变持久 Python 环境，并安装 `.[dev,training,evaluation]`。首次运行较慢；相同指纹的后续 Job 复用 venv、Hugging Face 缓存和 pip 缓存。

默认镜像包含约 3.1 GiB 的压缩主层，首次拉取可能需要十几分钟；本机实测完成镜像与全套依赖初始化后，第二个任务可直接输出“复用现有运行环境”。

修改 `pyproject.toml`、Python 版本或基础镜像后，工具会创建新指纹环境，不删除或原地更新正在运行的旧环境。并行提交多个首次任务时，文件锁保证只有一个任务安装依赖。

`--no-bootstrap` 会绕过 Python 项目入口，直接执行 `--` 后的命令，适用于 GPU 检查镜像或已经包含完整依赖的自定义镜像：

```bash
bash bash/k8s/gpu_job.sh submit \
  --name prebuilt \
  --gpus 1 \
  --image registry.example.com/c2c-runtime:cu124 \
  --no-bootstrap \
  -- python script/train/SFT_train.py --config <train-config>
```

使用默认 bootstrap 时，自定义镜像必须包含 Python 3.10+ 和 CUDA 12.4 兼容运行库。使用 `--no-bootstrap` 时，镜像只需包含用户请求的可执行命令，并能以当前宿主机 UID/GID 运行。

正式论文实验建议把 `--image` 写成带 digest 的不可变地址，避免远端同名 tag 更新后环境漂移。

## 故障排查

任务长时间 `Pending`：

```bash
bash bash/k8s/gpu_job.sh list
bash bash/k8s/gpu_job.sh describe <完整任务名>
kubectl get pods -n c2c-research -o wide
kubectl describe pod <pod-name> -n c2c-research
```

重点查看事件中的 `Insufficient nvidia.com/gpu`、CPU 或内存不足。资源释放后会自动调度；也可删除不再需要的 Job。

Pod 出现 `ImagePullBackOff`：

```bash
kubectl describe pod <pod-name> -n c2c-research
kubectl get events -n c2c-research --sort-by=.lastTimestamp
```

检查镜像地址、网络和仓库权限。私有仓库需先配置 `imagePullSecret`；当前启动器未自动注入凭据，可改用节点可访问的公共或内部镜像。

查看节点和 GPU 资源：

```bash
kubectl get node 4090-24gx4 -o wide
kubectl describe node 4090-24gx4
nvidia-smi
```

## 使用约束

Kubernetes 无法识别宿主机上直接运行的 `CUDA_VISIBLE_DEVICES=... python ...` 进程，可能把同一张 GPU 再分给 Pod。正式训练与评测统一通过本工具提交；使用 Kubernetes 时不要同时启动本地 GPU 进程。本地命令仅用于确认没有正式 Job 运行时的短时调试。
