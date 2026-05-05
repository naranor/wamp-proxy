import os
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_setfit_model():
    print("=== QUANTIZING SETFIT ONNX MODEL TO INT8 ===")
    model_dir = "./model_setfit_onnx"
    input_model = os.path.join(model_dir, "model.onnx")
    output_model = os.path.join(model_dir, "model_quantized.onnx")

    if not os.path.exists(input_model):
        print(f"Error: FP32 model not found at {input_model}")
        return

    print(f"Quantizing {input_model}...")

    # Dynamic quantization for transformer models
    quantize_dynamic(
        model_input=input_model, model_output=output_model, weight_type=QuantType.QInt8
    )

    # Calculate sizes
    size_fp32 = os.path.getsize(input_model) + os.path.getsize(input_model + ".data")
    size_int8 = os.path.getsize(output_model)

    print("\n✅ Quantization Complete!")
    print(f"FP32 Size: {size_fp32 / 1024 / 1024:.1f} MB")
    print(f"INT8 Size: {size_int8 / 1024 / 1024:.1f} MB")
    print(f"Reduction: {(1 - size_int8 / size_fp32) * 100:.1f}%")


if __name__ == "__main__":
    quantize_setfit_model()
