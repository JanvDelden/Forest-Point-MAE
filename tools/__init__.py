# from .runner import run_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_pretrain import test_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_cls as test_cls
from .runner_regression import run_net as regression 
from .runner_offset import test_offset as test_offset
from .runner_offset import run_net as offset
from .runner_segmentation import test_seg as test_seg
from .runner_segmentation import run_net as segmentation