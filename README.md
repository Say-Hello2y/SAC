# SAC算法实现
Soft Actor Critic 算法 pytorch 实现
## Getting Started
已在ubuntu20.04，22.04上测试过,复现步骤：

step1 :
```
pip install -r requirements.txt
```
step2(optional):（需注册wandb账号）
```
python train_sac.py --track
```
step3(不需要wandb账号):
```
python train_sac.py
```
## One More Thing
训练好的模型会保存到checkpoint目录，如果想用gym的mujoco环境请参考我知乎上的两篇文章[如何在ubuntu20.04上安装mujoco](https://zhuanlan.zhihu.com/p/535806578)，以及[如何在ubuntu20.04上配置pytorch GPU版本](https://zhuanlan.zhihu.com/p/535712148)。
