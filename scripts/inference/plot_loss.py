import re
import matplotlib.pyplot as plt

log_file = 'logs/yolo_train_4060103.out'
output_file = 'loss_curve.png'

epochs = []
box_losses = []
cls_losses = []
dfl_losses = []

with open(log_file, 'r') as f:
    lines = f.readlines()

# Regex to match the progress line when it hits 100%
# Example: 1/200      31.3G      2.127      4.362      2.387         70       1280: 100%
# Values correspond to header: Epoch GPU_mem box_loss cls_loss dfl_loss Instances Size
pattern = re.compile(r'\s*(\d+)/200\s+[\d\.]+G\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+.*100%')

for line in lines:
    match = pattern.search(line)
    if match:
        epoch = int(match.group(1))
        box_loss = float(match.group(2))
        cls_loss = float(match.group(3))
        dfl_loss = float(match.group(4))
        
        # Since 100% lines might appear multiple times or be weirdly logged, 
        # we'll store them and deduplicate by keeping the last one for each epoch
        # or just append and assume the file order is correct.
        # Simple approach: Store all, then filter unique epochs (last wins)
        epochs.append(epoch)
        box_losses.append(box_loss)
        cls_losses.append(cls_loss)
        dfl_losses.append(dfl_loss)

# Deduplicate: keep the last entry for each epoch
data = {}
for e, b, c, d in zip(epochs, box_losses, cls_losses, dfl_losses):
    data[e] = (b, c, d)

sorted_epochs = sorted(data.keys())
box_data = [data[e][0] for e in sorted_epochs]
cls_data = [data[e][1] for e in sorted_epochs]
dfl_data = [data[e][2] for e in sorted_epochs]

if not sorted_epochs:
    print("No complete epoch data found yet.")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(sorted_epochs, box_data, label='Box Loss')
plt.plot(sorted_epochs, cls_data, label='Cls Loss')
plt.plot(sorted_epochs, dfl_data, label='DFL Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('YOLO Training Loss')
plt.legend()
plt.grid(True)
plt.savefig(output_file)
print(f"Loss curve saved to {output_file}")
print(f"Parsed {len(sorted_epochs)} epochs.")
