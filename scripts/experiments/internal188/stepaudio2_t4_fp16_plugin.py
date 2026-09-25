"""Keep Step-Audio2's unquantized audio path in FP16 for T4 INT8 inference."""
import torch
from swift.model.models.stepfun import StepAudio2MiniLoader


if not hasattr(StepAudio2MiniLoader, '_daic_original_get_model'):
    StepAudio2MiniLoader._daic_original_get_model = StepAudio2MiniLoader.get_model

    def get_model_t4(self, model_dir, *args, **kwargs):
        model = StepAudio2MiniLoader._daic_original_get_model(self, model_dir, *args, **kwargs)
        for name in ('encoder', 'adapter', 'lm_head'):
            getattr(model, name).to(dtype=torch.float16)

        def cast_encoder_input(module, args):
            if not args or args[0] is None: return args
            dtype = next(module.parameters()).dtype
            return (args[0].to(dtype=dtype), *args[1:])

        model.encoder.register_forward_pre_hook(cast_encoder_input)
        print('[DAIC Step-Audio2 T4] encoder input hook installed; model.forward unchanged', flush=True)
        return model

    StepAudio2MiniLoader.get_model = get_model_t4
