import onnx
import onnx.helper as helper
import numpy as np

onnx_file = "../models/yolov5s-test.onnx"

model = onnx.load(onnx_file)

for item in model.graph.initializer:
    if item.name=="456":
        print(item.name)
        print(item.dims)
        print(item.data_type)
        print(item)
        print(np.frombuffer(item.raw_data, dtype=np.int64))

# print(model.graph.input)

for node in model.graph.node:
    if node.op_type == "Reshape":
        print(node.name)
        print(node.op_type)
        print(node)



# reshape_nodes = [node for node in model.nodes if node.op == "Reshape"]
# for node in reshape_nodes:
#     # The batch dimension in the input shape is hard-coded to a static value in the original model.
#     # To make the model work with our dynamic batch size, we can use a `-1`, which indicates that the
#     # dimension should be automatically determined.
#     print(node)
#     node.inputs[1].values[0] = -1