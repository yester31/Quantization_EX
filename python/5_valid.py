from utils import *

output_c = np.fromfile("data/trt_output.bin", dtype=np.float32)
output_py = np.fromfile("data/torch_output.bin", dtype=np.float32)
compare_two_tensor(output_py, output_c)