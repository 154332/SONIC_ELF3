from pathlib import Path
import numbers
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

class TensorBoardCallback(TrainerCallback):
    def __init__(self,log_dir):
        super().__init__()
        self.log_dir=Path(log_dir)
        self.writer=None
    def _get_writer(self):
        if self.writer is None:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            self.writer=SummaryWriter(log_dir=str(self.log_dir))
        return self.writer
    def on_log(self,args,state,control,logs=None,**kwargs):
        if not state.is_world_process_zero or not logs:
            return
        writer=self._get_writer()
        for key,value in logs.items():
            if hasattr(value,"item"):
                try:
                    value=value.item()
                except Exception:
                    continue
            if isinstance(value,numbers.Number):
                writer.add_scalar(key,value,state.global_step)
        writer.flush()
    def on_train_end(self,args,state,control,**kwargs):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
