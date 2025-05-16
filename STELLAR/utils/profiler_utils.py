import os
import torch
import transformers
from torch.autograd.profiler import record_function

class ProfilerCallback(transformers.TrainerCallback):
    """在训练循环的不同阶段启用Profiler记录"""
    
    def __init__(self, profiler):
        self.profiler = profiler
    
    def on_step_begin(self, args, state, control, **kwargs):
        """每个step开始时调用profiler.step()"""
        self.profiler.step()
        
    def on_prediction_step(self, args, state, control, **kwargs):
        """记录前向传播阶段"""
        with record_function("## forward ##"):
            pass
            
    def on_backward(self, args, state, control, **kwargs):
        """记录反向传播阶段"""
        with record_function("## backward ##"):
            pass
            
    def on_step_end(self, args, state, control, **kwargs):
        """记录优化器步骤阶段"""
        with record_function("## optimizer_step ##"):
            pass

class ProfilerTrainer(transformers.Trainer):
    """支持Profiler的自定义Trainer"""
    
    def __init__(self, profiler=None, **kwargs):
        super().__init__(**kwargs)
        self.profiler = profiler
        if self.profiler is not None:
            self.add_callback(ProfilerCallback(self.profiler))
            
    def train(self, **kwargs):
        """重写train方法以支持profiler上下文"""
        if self.profiler is not None:
            with self.profiler:
                return super().train(**kwargs)
        else:
            return super().train(**kwargs)