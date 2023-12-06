**News!!!**   
**[12/03/2023]** Manuscript `Accelerating Heterogeneous Tensor Parallelism via Flexible Workload Control` is submitted to ICDE 2024 for under review!!!  
**[10/21/2023]** Some bugs about priority pruning is solved.   
**[09/16/2023]** SEMI-migration is available on top of resizing and migration.   
**[07/29/2023]** Data migration is online.  
**[06/29/2013]** Priority pruning is online.  
**[06/15/2013]** ViT with 1.2B and 2.7B parameters is successfully run.  
**[05/24/2023]** ZERO-resizing/pruning technique is implemented and its effectiveness is validated.   
**[03/24/2023]** Colossal-AI is succefully deployed and tested on 8 V100 GPUs.  
**[03/12/2023]** This project starts.  

# FlexTP
FlexTP consists of a series of optimization techniques tailored for efficient Tensor Parallelism in heterogeneous environments. It is particularly suitable for training foundation models with billions of parameters.    

## 1. Introduction
Recently, transformer-based models are becoming deeper and larger. For better scalability, an underlying training solution is to split billions of parameters into many tasks for parallel computation, i.e., tensor parallelism. The computing cluster, however, usually consists of heterogeneous devices, and is shared by multi-tenants. The heterogeneous compute power and resource contention lead to the heavy straggling problem. Existing solutions for it are all tailored for traditional data parallelism. That cannot work under tensor parallelism due to the correctness and convergence constraint. `FlexTP` is the first work that attempts to flexibly control workloads for balanced tensor parallelism.  

The FlexTP project started at Ocean University of China in March 2023. The backbone members are **Xu Zhang** (master student) and **Zhigang Wang** (associate professor). We implement all related techniques on the newly released platform Colossal-AI, which has integrated many existing optimizations for training foundation models. Features of Flex-TP are listed as below.   

* ___ZERO-resizing:___ We temporarily resize matrices involved in core tensor computations to dynamically balance workloads, which employs data imputation and priority selection policies to guarantee consistency in computations and mitigate the loss of accuracy.  
* ___Lightweight migration:___ We re-assign matrix data to cope with heavily heterogeneous environments without accuracy loss, which reduces the migration runtime latency via existing tree-based broadcasting/reducing efforts and our novel reducing merging optimization.    
* ___SEMI-migration:___ We build this hybrid solution on top of the two approaches, which can smartly run the reasonable one in different scenarios based on the cost-benefit analysis.   

## 2. Quick Start
FlexTP is developed on top of Colossal-AI. Before running it, some softwares must be installed, which is beyond the scope of this document. 

### 2.1 Requirements
* Colossal-AI 0.2.7  
* PyTorch >= 1.13
* Python 3.7 or higher version   

### 2.2 Testing FlexTP  
We should first prepare the input training samples.     

Second, we can submit a training job for two foundation models using the following commands.  (The config file is stored in ./examples/hybrid_parallel)
* __For ViT-1B(1.2B):__  
`colossalai run --nproc_per_node 8 timer_trainer_with_cifar10.py --config tpconfig.py`  
About arguments:  
[1] GPUs num
[2] trainer file
[3] config file

* __For ViT-3B(2.7B):__  
`colossalai run --nproc_per_node 8 timer_trainer_with_cifar10.py --config finalconfig.py`  
About arguments:  
[1] GPUs num
[2] trainer file
[3] config file


## 3. Contact  
If you encounter any problem with FlexTP, please feel free to contact zhangxu1126@stu.ouc.edu.cn and wangzhigang@ouc.edu.cn.

