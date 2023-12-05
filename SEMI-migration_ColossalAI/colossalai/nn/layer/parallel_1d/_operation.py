import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils.cuda import get_current_device
import random
import time
import torch.nn.functional as F

try:
    import fused_mix_prec_layer_norm_cuda
except:
    fused_mix_prec_layer_norm_cuda = None


class FusedLayerNormAffineFunction1D(torch.autograd.Function):
    r"""Layernorm

    Args:
        input: input matrix.
        weight: weight matrix.
        bias: bias matrix.
        normalized_shape: input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1] \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability
  """

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(input_, ctx.normalized_shape, weight_,
                                                                             bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias \
          = fused_mix_prec_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar,
            input_, ctx.normalized_shape,
            weight_, bias_, ctx.eps)

        return grad_input, grad_weight, grad_bias, None, None


class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag, migration_list, compute_list):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.priority_list = priority_list
        ctx.cur_epoch = cur_epoch
        ctx.eval_flag = eval_flag
        ctx.migration_list = migration_list
        ctx.compute_list = compute_list
        
        device = torch.cuda.current_device()
        
        #zx:异构卡广播      
        #randint_data = random.randint(0,7)
        #logger = get_dist_logger()
        
        copy_weight0 = weight.clone()
        copy_mig_list0 = torch.arange(int(weight.shape[1]/32)*4*7, device=device)

        if device==0 or cur_epoch==0:
            copy_mig_list0 = migration_list.clone()
        
        dist.broadcast(copy_weight0, src=0, group=gpc.get_group(ctx.parallel_mode))
        dist.broadcast(copy_mig_list0, src=0, group=gpc.get_group(ctx.parallel_mode))

        copy_mig_list0 = torch.chunk(copy_mig_list0, chunks=4, dim=0)
        mig_device_num = device-4
        
        #cur_epoch%8 == device:
        #zx:中间tensor填充
        if device==0 and eval_flag==1:
        #if cur_epoch%8 != 10:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())

            mig_output0 = output
            mig_output1 = torch.zeros_like(output)
        elif device==1 and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            
            mig_output0 = torch.zeros_like(output)
        elif device==2 and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())

            mig_output0 = torch.zeros_like(output)
        elif device==3 and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())

            mig_output0 = torch.zeros_like(output)
        else:
            output = torch.matmul(input_, weight.t())
            #if bias is not None:
                #output = output + bias
            #logger.info(f'time2: {end2-start2} {device}',ranks=[0,1])
            
            tmp_input0 = torch.index_select(input_,dim=2,index=copy_mig_list0[mig_device_num])
            tmp_weight0 = torch.index_select(copy_weight0,dim=1,index=copy_mig_list0[mig_device_num])
            mig_output0 = torch.matmul(tmp_input0, tmp_weight0.t())

        dist.reduce(mig_output0, dst=0, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        if device==0 and eval_flag==1:
            output = mig_output0
        
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        priority_list = ctx.priority_list
        cur_epoch = ctx.cur_epoch
        eval_flag = ctx.eval_flag
        migration_list =ctx.migration_list
        compute_list = ctx.compute_list

        total_input = input_
        
        #zx:异构卡广播      
        device = torch.cuda.current_device()

        copy_weight0 = weight.clone()
        copy_mig_list0 = torch.arange(int(weight.shape[1]/32)*4*7, device=device)
        copy_grad_output0 = grad_output.clone()

        if device==0 or cur_epoch==0:
            copy_mig_list0 = migration_list.clone()
        
        
        dist.broadcast(copy_weight0, src=0, group=gpc.get_group(ctx.parallel_mode))
        dist.broadcast(copy_mig_list0, src=0, group=gpc.get_group(ctx.parallel_mode))
        dist.broadcast(copy_grad_output0, src=0, group=gpc.get_group(ctx.parallel_mode))

        copy_mig_list0 = torch.chunk(copy_mig_list0, chunks=4, dim=0)
        mig_device_num = device-4

        #zx：梯度截取
        #logger = get_dist_logger()
        #logger.info(f'copy_mig_list: {copy_mig_list.shape} {device}', ranks=[0,1,2,3,4,5,6,7])
        if device==0:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
        elif device==1:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
        elif device==2:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
        elif device==3:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)        
            
        else:
            grad_input = grad_output.matmul(weight)
            
            tmp_weight0 = torch.index_select(copy_weight0,dim=1,index=copy_mig_list0[mig_device_num])
            tmp_grad_input0 = copy_grad_output0.matmul(tmp_weight0)
            grad_input.index_add_(2, copy_mig_list0[mig_device_num], tmp_grad_input0)
        '''
        dist.reduce(g_input, cur_epoch%8, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        if cur_epoch%8 == device:
            grad_input = g_input
        '''
        
        #logger = get_dist_logger()
        #logger.info(f'in_shape1: {grad_input.shape}', ranks=[0])
        #logger.info(f'out_shape1: {grad_output.shape}', ranks=[0,1])

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=gpc.get_group(ctx.parallel_mode), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        
        if device==0:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)  

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)

            mig_grad_weight0 = grad_weight.clone()
        elif device==1:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)  

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)

            mig_grad_weight0 = torch.zeros_like(grad_weight)
        elif device==2:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)

            mig_grad_weight0 = torch.zeros_like(grad_weight)
        elif device==3:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            
            mig_grad_weight0 = torch.zeros_like(grad_weight)
                    
        else:
            grad_weight = grad_output.t().matmul(total_input)

            tmp_input0 = torch.index_select(total_input,dim=1,index=copy_mig_list0[mig_device_num])
            copy_grad_output0 = copy_grad_output0.view(copy_grad_output0.shape[0] * copy_grad_output0.shape[1], copy_grad_output0.shape[2])
            tmp_grad_weight0 = copy_grad_output0.t().matmul(tmp_input0)
            mig_grad_weight0 = torch.zeros_like(grad_weight)
            mig_grad_weight0.index_add_(1, copy_mig_list0[mig_device_num], tmp_grad_weight0)

        
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        '''
        if cur_epoch%8 == device:
            #gather_w_list = [torch.zeros((x_num,copy_mig_list[(i+8-cur_epoch%8)%8-1].shape[0]),device=device) for i in range(8)]
            gather_w_list = [torch.zeros_like(mig_grad_weight) for i in range(8)]
            #gather_w_list = torch.zeros_like(tmp_grad_weight)
            dist.gather(mig_grad_weight, gather_w_list, dst=cur_epoch%8, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        else:
            dist.gather(mig_grad_weight, dst=cur_epoch%8, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        
        if cur_epoch%8 == device:
            for mig_d_num in range(len(gather_w_list)):
                if mig_d_num != device:
                    tmp_grad_weight = gather_w_list[mig_d_num]
                    grad_weight.index_add_(1, copy_mig_list[(mig_d_num+8-cur_epoch%8)%8-1], tmp_grad_weight)
        '''
        dist.reduce(mig_grad_weight0, dst=0, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        
        if device==0:
            grad_weight = mig_grad_weight0

        if ctx.async_grad_allreduce:
            handle.wait() 
        
        '''
        if cur_epoch > -1:
            logger = get_dist_logger()
            #logger.info(f'tmp_grad_input: {tmp_grad_input[0]} {tmp_grad_input[0].shape} {device}', ranks=[0,1,7])
            logger.info(f'grad_input: {grad_input[0]} {grad_input[0].shape} {device}', ranks=[0,1])
        '''
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class RowcutLinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag, migration_list, compute_list):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.priority_list = priority_list
        ctx.cur_epoch = cur_epoch
        ctx.eval_flag = eval_flag
        ctx.migration_list = migration_list
        ctx.compute_list = compute_list
        
        device = torch.cuda.current_device()
        
        #zx:异构卡广播      
        #randint_data = random.randint(0,7)
        #logger = get_dist_logger()
        
        copy_weight0 = weight.clone()
        copy_mig_list0 = torch.arange(int(weight.shape[1]/32)*4*7, device=device)

        if device==0 or cur_epoch==0:
            copy_mig_list0 = migration_list.clone()
        
        dist.broadcast(copy_weight0, src=0, group=gpc.get_group(ctx.parallel_mode))
        dist.broadcast(copy_mig_list0, src=0, group=gpc.get_group(ctx.parallel_mode))

        copy_mig_list0 = torch.chunk(copy_mig_list0, chunks=4, dim=0)
        mig_device_num = device-4
        
        #cur_epoch%8 == device:
        #zx:中间tensor填充
        if device==0 and eval_flag==1:
        #if cur_epoch%8 != 10:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())

            mig_output0 = output
            mig_output1 = torch.zeros_like(output)
        elif device==1 and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            
            mig_output0 = torch.zeros_like(output)
        elif device==2 and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())
            output22 = torch.matmul(tmp_input, tmp_weight.t())

            mig_output0 = torch.zeros_like(output)
        elif device==3 and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=compute_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)           
            output = torch.matmul(tmp_input, tmp_weight.t())

            output22 = torch.matmul(tmp_input, tmp_weight.t())

            mig_output0 = torch.zeros_like(output)
        else:
            output = torch.matmul(input_, weight.t())
            #if bias is not None:
                #output = output + bias
            #logger.info(f'time2: {end2-start2} {device}',ranks=[0,1])
            
            tmp_input0 = torch.index_select(input_,dim=2,index=copy_mig_list0[mig_device_num])
            tmp_weight0 = torch.index_select(copy_weight0,dim=1,index=copy_mig_list0[mig_device_num])
            mig_output0 = torch.matmul(tmp_input0, tmp_weight0.t())

        dist.reduce(mig_output0, dst=0, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        if device==0 and eval_flag==1:
            output = mig_output0
        
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        priority_list = ctx.priority_list
        cur_epoch = ctx.cur_epoch
        eval_flag = ctx.eval_flag
        migration_list =ctx.migration_list
        compute_list = ctx.compute_list

        total_input = input_
        
        #zx:异构卡广播      
        device = torch.cuda.current_device()

        copy_weight0 = weight.clone()
        copy_mig_list0 = torch.arange(int(weight.shape[1]/32)*4*7, device=device)
        copy_grad_output0 = grad_output.clone()

        if device==0 or cur_epoch==0:
            copy_mig_list0 = migration_list.clone()
        
        
        dist.broadcast(copy_weight0, src=0, group=gpc.get_group(ctx.parallel_mode))
        dist.broadcast(copy_mig_list0, src=0, group=gpc.get_group(ctx.parallel_mode))
        dist.broadcast(copy_grad_output0, src=0, group=gpc.get_group(ctx.parallel_mode))

        copy_mig_list0 = torch.chunk(copy_mig_list0, chunks=4, dim=0)
        mig_device_num = device-4

        #zx：梯度截取
        #logger = get_dist_logger()
        #logger.info(f'copy_mig_list: {copy_mig_list.shape} {device}', ranks=[0,1,2,3,4,5,6,7])
        if device==0:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
        elif device==1:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
        elif device==2:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
            tmp_grad_input22 = grad_output.matmul(tmp_weight)
        elif device==3:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=compute_list)
            #tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=compute_list)
            
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            descend_num = torch.full((x_num,y_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            #grad_input.index_add_(2, priority_list, tmp_grad_input)
            grad_input.index_add_(2, compute_list, tmp_grad_input)
            '''
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input = torch.add(grad_input, 1e-18, alpha=-1) must be specified on the destination rank
            grad_input.index_add_(2, copy_pri_list, tmp_grad_input)
            '''
            tmp_grad_input22 = grad_output.matmul(tmp_weight)        
            
        else:
            grad_input = grad_output.matmul(weight)
            
            tmp_weight0 = torch.index_select(copy_weight0,dim=1,index=copy_mig_list0[mig_device_num])
            tmp_grad_input0 = copy_grad_output0.matmul(tmp_weight0)
            grad_input.index_add_(2, copy_mig_list0[mig_device_num], tmp_grad_input0)
        '''
        dist.reduce(g_input, cur_epoch%8, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        if cur_epoch%8 == device:
            grad_input = g_input
        '''
        
        #logger = get_dist_logger()
        #logger.info(f'in_shape1: {grad_input.shape}', ranks=[0])
        #logger.info(f'out_shape1: {grad_output.shape}', ranks=[0,1])

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=gpc.get_group(ctx.parallel_mode), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        
        if device==0:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)  

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)

            mig_grad_weight0 = grad_weight.clone()
        elif device==1:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)  

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)

            mig_grad_weight0 = torch.zeros_like(grad_weight)
        elif device==2:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            tmp_grad_weight22 = grad_output.t().matmul(total_input)

            mig_grad_weight0 = torch.zeros_like(grad_weight)
        elif device==3:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)
            descend_num = torch.full((x_num,priority_list.shape[0]), fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, compute_list, tmp_grad_weight)  
            #mig_grad_weight = torch.zeros((x_num,copy_mig_list[0].shape[0]),device=device)

            tmp_grad_weight22 = grad_output.t().matmul(total_input)
            
            mig_grad_weight0 = torch.zeros_like(grad_weight)
                    
        else:
            grad_weight = grad_output.t().matmul(total_input)

            tmp_input0 = torch.index_select(total_input,dim=1,index=copy_mig_list0[mig_device_num])
            copy_grad_output0 = copy_grad_output0.view(copy_grad_output0.shape[0] * copy_grad_output0.shape[1], copy_grad_output0.shape[2])
            tmp_grad_weight0 = copy_grad_output0.t().matmul(tmp_input0)
            mig_grad_weight0 = torch.zeros_like(grad_weight)
            mig_grad_weight0.index_add_(1, copy_mig_list0[mig_device_num], tmp_grad_weight0)

        
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        '''
        if cur_epoch%8 == device:
            #gather_w_list = [torch.zeros((x_num,copy_mig_list[(i+8-cur_epoch%8)%8-1].shape[0]),device=device) for i in range(8)]
            gather_w_list = [torch.zeros_like(mig_grad_weight) for i in range(8)]
            #gather_w_list = torch.zeros_like(tmp_grad_weight)
            dist.gather(mig_grad_weight, gather_w_list, dst=cur_epoch%8, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        else:
            dist.gather(mig_grad_weight, dst=cur_epoch%8, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        
        if cur_epoch%8 == device:
            for mig_d_num in range(len(gather_w_list)):
                if mig_d_num != device:
                    tmp_grad_weight = gather_w_list[mig_d_num]
                    grad_weight.index_add_(1, copy_mig_list[(mig_d_num+8-cur_epoch%8)%8-1], tmp_grad_weight)
        '''
        dist.reduce(mig_grad_weight0, dst=0, group=gpc.get_group(ctx.parallel_mode), async_op=False)
        
        if device==0:
            grad_weight = mig_grad_weight0

        if ctx.async_grad_allreduce:
            handle.wait() 
        
        '''
        if cur_epoch > -1:
            logger = get_dist_logger()
            #logger.info(f'tmp_grad_input: {tmp_grad_input[0]} {tmp_grad_input[0].shape} {device}', ranks=[0,1,7])
            logger.info(f'grad_input: {grad_input[0]} {grad_input[0].shape} {device}', ranks=[0,1])
        '''
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None
        


def linear_with_async_comm(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag, migration_list, compute_list):
    return LinearWithAsyncCommunication.apply(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag, migration_list, compute_list)

def rowcut_linear_with_async_comm(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag, migration_list, compute_list):
    return RowcutLinearWithAsyncCommunication.apply(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag, migration_list, compute_list)
