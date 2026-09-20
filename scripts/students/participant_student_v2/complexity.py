from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,torch
from torch.profiler import profile,ProfilerActivity

def analytical_macs(segments,text_dim=250):
    # Exact v2 forward MACs for one participant. Bias/normalization/activations are omitted,
    # as is conventional in MAC reporting.
    l1=192; l2=96; d=96; h=64
    conv1=l1*d*130*5
    depthwise=l2*d*5
    pointwise=l2*d*d
    gru=l2*3*(d*h+h*h)
    projection=h*d
    audio=int(segments)*(conv1+depthwise+pointwise+gru+projection)
    graph=text_dim*text_dim
    text_mlp=(2*text_dim)*128+128*d
    cross=int(segments)*(d*d+d)+d*d
    classifier=(4*d)*128+128*32+32
    return int(audio+graph+text_mlp+cross+classifier)

def profile_model(model,audio,text,mask,device,checkpoint=None,repeats=50,warmup=10):
    model=model.to(device).eval(); audio=audio.to(device); text=text.to(device); mask=mask.to(device)
    with torch.inference_mode():
        for _ in range(warmup): model(audio,text,mask)
    if device.type=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    acts=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device.type=="cuda" else [])
    with profile(activities=acts,with_flops=True,record_shapes=False) as prof:
        with torch.inference_mode(): model(audio,text,mask)
    flops=int(sum(int(getattr(e,"flops",0) or 0) for e in prof.key_averages()))
    times=[]
    with torch.inference_mode():
        for _ in range(repeats):
            if device.type=="cuda": torch.cuda.synchronize()
            t=time.perf_counter(); model(audio,text,mask)
            if device.type=="cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
    params=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs=analytical_macs(int(audio.shape[1]),int(text.shape[1]))
    return {"total_parameters":int(params),"trainable_parameters":int(trainable),
            "checkpoint_size_mb":Path(checkpoint).stat().st_size/1024**2 if checkpoint and Path(checkpoint).exists() else None,
            "profile_input":{"batch":int(audio.shape[0]),"audio_segments":int(audio.shape[1]),"compare16_bins":int(audio.shape[2]),
                             "frames_per_segment":int(audio.shape[3]),"text_tfidf_dim":int(text.shape[1])},
            "analytical_macs":macs,"analytical_gmacs":float(macs/1e9),
            "analytical_flops_2_per_mac":int(2*macs),"analytical_gflops_2_per_mac":float(2*macs/1e9),
            "analytical_method":"Conv/GRU/Linear matmul MACs; normalization, bias, activations and softmax omitted; 2 FLOPs per MAC",
            "profiled_flops":flops,"profiled_gflops":float(flops/1e9),
            "profiler_method":"torch.profiler with_flops=True; supported PyTorch operators only (may omit recurrent kernels)",
            "latency_ms_mean":float(np.mean(times)),"latency_ms_std":float(np.std(times)),
            "throughput_participants_per_sec":float(1000/np.mean(times)),
            "peak_cuda_memory_mb":float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else None,
            "device":str(device)}

def save(path,obj): Path(path).write_text(json.dumps(obj,indent=2)+"\n")
