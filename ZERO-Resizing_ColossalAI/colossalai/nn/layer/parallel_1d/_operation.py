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

'''
class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, parallel_mode, async_grad_allreduce, cur_epoch):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.cur_epoch = cur_epoch
        
        device = torch.cuda.current_device()

        #zx:记录时间
        #logger = get_dist_logger()
        #logger.info(f'time1: {end1-start1} {device}',ranks=[0])
        torch.cuda.synchronize(device=device)
        start1 = time.time()
        output = torch.matmul(input_, weight.t())
        torch.cuda.synchronize(device=device)
        end1 = time.time()
        
        if bias is not None:
            output = output + bias
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        cur_epoch = ctx.cur_epoch

        device = torch.cuda.current_device()

        total_input = input
        torch.cuda.synchronize(device=device)
        start1 = time.time()
        grad_input = grad_output.matmul(weight)
        torch.cuda.synchronize(device=device)
        end1 = time.time()
        compute_time = end1 - start1

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=gpc.get_group(ctx.parallel_mode), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        torch.cuda.synchronize(device=device)
        start2 = time.time()
        grad_weight = grad_output.t().matmul(total_input)
        torch.cuda.synchronize(device=device)
        end2 = time.time()
        compute_time = compute_time + end2 - start2
        
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()
        
        return grad_input, grad_weight, grad_bias, None, None, None, None
'''

class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.priority_list = priority_list
        ctx.cur_epoch = cur_epoch
        ctx.eval_flag = eval_flag

        #zx:中间tensor填充
        #randint_data = random.randint(0,7)
        #logger = get_dist_logger()
        device = torch.cuda.current_device()

        #cur_epoch%8 == device:
        
        if cur_epoch%8 == device and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            output = torch.matmul(tmp_input, tmp_weight.t())

            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            
            if bias is not None:
                output = output + bias
        else:
            output = torch.matmul(input_, weight.t())
            if bias is not None:
                output = output + bias
            #logger.info(f'time2: {end2-start2} {device}',ranks=[0,1])
        '''
        delay_time = (end1 - start1) * 1
        if cur_epoch%8 == device:
            time.sleep(delay_time)   
        '''

        '''
        if cur_epoch > 0:
            logger = get_dist_logger()
            logger.info(f'time01: {end1-start1} {device}',ranks=[0,1])     
        
        ''' 
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        priority_list = ctx.priority_list
        cur_epoch = ctx.cur_epoch
        eval_flag = ctx.eval_flag

        total_input = input_

        #zx：梯度截取
        device = torch.cuda.current_device()

        #logger = get_dist_logger()
        #logger.info(f'out_shape2: {grad_output}', ranks=[1])
        if cur_epoch%8 == device:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #fill_num = input_num - cut_num
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            
            descend_num = torch.full_like(tmp_grad_input, fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            grad_input.index_add_(2, priority_list, tmp_grad_input)
        else:
            grad_input = grad_output.matmul(weight)

        #logger = get_dist_logger()
        #logger.info(f'in_shape1: {grad_input.shape}', ranks=[0])
        #logger.info(f'out_shape1: {grad_output.shape}', ranks=[0,1])

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        
        '''
        if device == 0:
            sleep_time = 100 * compute_time
            time.sleep(sleep_time)
        '''
        
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=gpc.get_group(ctx.parallel_mode), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        
        if cur_epoch%8 == device:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)

            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            
            descend_num = torch.full_like(tmp_grad_weight, fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, priority_list, tmp_grad_weight)            
        else:
            grad_weight = grad_output.t().matmul(total_input)
        
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        '''  
        delay_time = compute_time * 1
        if cur_epoch%8 == device:
            time.sleep(delay_time)   
        '''
        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class RowcutLinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.priority_list = priority_list
        ctx.cur_epoch = cur_epoch
        ctx.eval_flag = eval_flag

        #zx:中间tensor填充
        #randint_data = random.randint(0,7)
        #logger = get_dist_logger()
        device = torch.cuda.current_device()

        #cur_epoch%8 == device:
        
        if cur_epoch%8 == device and eval_flag==1:
            tmp_input = torch.index_select(input_,dim=2,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            output = torch.matmul(tmp_input, tmp_weight.t())

            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            output2 = torch.matmul(tmp_input, tmp_weight.t())
            
            if bias is not None:
                output = output + bias
        else:
            output = torch.matmul(input_, weight.t())
            if bias is not None:
                output = output + bias
            #logger.info(f'time2: {end2-start2} {device}',ranks=[0,1])
        '''
        delay_time = (end1 - start1) * 1
        if cur_epoch%8 == device:
            time.sleep(delay_time)   
        '''

        '''
        if cur_epoch > 0:
            logger = get_dist_logger()
            logger.info(f'time01: {end1-start1} {device}',ranks=[0,1])     
        
        ''' 
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        priority_list = ctx.priority_list
        cur_epoch = ctx.cur_epoch
        eval_flag = ctx.eval_flag

        total_input = input_

        #zx：梯度截取
        device = torch.cuda.current_device()

        #logger = get_dist_logger()
        #logger.info(f'out_shape2: {grad_output}', ranks=[1])
        if cur_epoch%8 == device:
            input_num = total_input.shape[2]
            weight_num = weight.shape[1]
            #cut_num = int(input_num/2)
            #fill_num = input_num - cut_num
            #total_input = torch.narrow(total_input, 2, 0, cut_num)
            #tmp_weight = torch.narrow(weight, 1, 0, cut_num)
            fill_num = weight_num - priority_list.shape[0]
            total_input = torch.index_select(total_input,dim=2,index=priority_list)
            tmp_weight = torch.index_select(weight,dim=1,index=priority_list)
            #logger = get_dist_logger()
            #logger.info(f'pri_shape1: {priority_list.shape}', ranks=[0])
            #logger.info(f'num: {fill_num}', ranks=[0])
            #grad_input = grad_output.matmul(tmp_weight)

            #zx:离散优先级
            x_num = grad_output.shape[0]
            y_num = grad_output.shape[1]
            z_num = weight.shape[1]
            tmp_grad_input = grad_output.matmul(tmp_weight)

            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            tmp_grad_input2 = grad_output.matmul(tmp_weight)
            
            descend_num = torch.full_like(tmp_grad_input, fill_value=1e-18, device=device)
            grad_input = torch.full((x_num,y_num,z_num), fill_value=1e-18, device=device)
            grad_input.index_add_(2, priority_list, descend_num, alpha=-1)
            grad_input.index_add_(2, priority_list, tmp_grad_input)
            '''
            if cur_epoch > 1:
                logger = get_dist_logger()
                logger.info(f'shape: {priority_list.shape[0]} {device}', ranks=[device])
            '''
        else:
            grad_input = grad_output.matmul(weight)

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
        
        if cur_epoch%8 == device:
            #zx:离散优先级
            x_num = grad_output.shape[1]
            y_num = input_.shape[2]
            tmp_grad_weight = grad_output.t().matmul(total_input)

            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            tmp_grad_weight2 = grad_output.t().matmul(total_input)
            
            descend_num = torch.full_like(tmp_grad_weight, fill_value=1e-18, device=device)
            grad_weight = torch.full((x_num,y_num), fill_value=1e-18, device=device)
            grad_weight.index_add_(1, priority_list, descend_num, alpha=-1)
            grad_weight.index_add_(1, priority_list, tmp_grad_weight)            
        else:
            grad_weight = grad_output.t().matmul(total_input)
        
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        '''  
        delay_time = compute_time * 1
        if cur_epoch%8 == device:
            time.sleep(delay_time)   
        '''
        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
        


def linear_with_async_comm(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag):
    return LinearWithAsyncCommunication.apply(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag)

def rowcut_linear_with_async_comm(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag):
    return RowcutLinearWithAsyncCommunication.apply(input_, weight, bias, parallel_mode, async_grad_allreduce, priority_list, cur_epoch, eval_flag)
