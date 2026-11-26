import sys, time, torch
import PyNvVideoCodec as nvc
path = sys.argv[1]
print([n for n in dir(nvc) if not n.startswith('_')])
dec = nvc.SimpleDecoder(path, gpu_id=0, use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP)
md = dec.get_stream_metadata()
print(md)
t=time.time(); n=0
for i in range(0, len(dec), 64):
    frames = dec.get_batch_frames(64)
    if not frames: break
    for f in frames:
        x = torch.from_dlpack(f)
        n+=1
    if n==64: print(x.shape, x.dtype, x.device)
torch.cuda.synchronize(); print(n, n/(time.time()-t), "fps")
