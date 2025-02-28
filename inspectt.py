import onnx

model = onnx.load("emotion_model.onnx")  # or your gender/age model
print("Opset Imports:", model.opset_import)
for imp in model.opset_import:
    print("Domain:", imp.domain, "Version:", imp.version)

print("----- Operators in the model -----")
for node in model.graph.node:
    print(node.op_type)
