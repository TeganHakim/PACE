import torch

from reliability_gate import ReliabilityGate


gate = ReliabilityGate()

# Create two quality maps.
# Start with everything low quality.
rgb_quality = torch.rand(2, 1, 8, 8)
ir_quality = torch.rand(2, 1, 8, 8)

# Image 0:
# Left half is high quality, right half is low quality.
rgb_quality[0, :, :, :4] = 1.0
ir_quality[0, :, :, :4] = 1.0

# Image 1:
# Top half is high quality, bottom half is low quality.
rgb_quality[1, :, :4, :] = 1.0
ir_quality[1, :, :4, :] = 1.0


# Run the gate.
output = gate(rgb_quality, ir_quality)


# Check shape.
print("Output shape:", output.shape)


# Print binary gate matrices.
for i in range(output.shape[0]):
    print(f"\nGate {i}:")
    
    for row in output[i, 0].int():
        print(" ".join(map(str, row.tolist())))