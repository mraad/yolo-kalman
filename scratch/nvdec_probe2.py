import sys, time, torch, threading
import PyNvVideoCodec as nvc
paths = sys.argv[2:]; nthreads=int(sys.argv[1])
torch.cuda.init()
def run(path, gpu, res):
    torch.cuda.set_device(gpu)
    dec = nvc.ThreadedDecoder(path, buffer_size=32, gpu_id=gpu, use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP)
    n=0; s=0
    while True:
        frames = dec.get_batch_frames(16)
        if len(frames)==0: break
        for f in frames:
            x = torch.from_dlpack(f); s += int(x[0,0,0]) if n%1000==0 else 0; n+=1
    res.append(n)
t=time.time(); res=[]
th=[threading.Thread(target=run,args=(paths[i%len(paths)], i%2, res)) for i in range(nthreads)]
[x.start() for x in th]; [x.join() for x in th]
dt=time.time()-t; print(nthreads, "decoders", sum(res), "frames", sum(res)/dt, "fps total")
