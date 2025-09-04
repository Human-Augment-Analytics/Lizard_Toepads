from ultralytics import YOLO

model = YOLO("yolo11s.pt")  

# this configuration assumes that 2 H200 GPUs have been allocated
model.train(
    data="data/data.yaml",
    epochs=100,
    batch=32,           # increase batch size significantly
    imgsz=1024,         # match resized images
    workers=8,          # more workers for data loading
    patience=20,
    name="tps_yolo_exp_h200_multi",
    device=[0, 1],      
    amp=True,           # mixed precision for H200s
    cache=True,         # cache images in RAM for faster loading
    save_period=10,     # save checkpoints every 10 epochs
) # we can configure all of these btw 