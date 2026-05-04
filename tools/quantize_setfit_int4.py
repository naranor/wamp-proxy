import os
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig
import logging

def quantize_setfit_int4():
    print("=== QUANTIZING SETFIT ONNX MODEL TO INT4 (MATMUL-NBITS) ===")
    model_dir = "./model_setfit_onnx"
    input_model_path = os.path.join(model_dir, "model.onnx")
    output_model_path = os.path.join(model_dir, "model_int4.onnx")
    
    if not os.path.exists(input_model_path):
        print(f"Error: FP32 model not found at {input_model_path}")
        return

    # 1. Load the FP32 ONNX model
    print(f"Loading {input_model_path}...")
    model = onnx.load(input_model_path)

    # 2. Configure Weight-Only Quantization
    # bits: 4
    # block_size: 32 (better accuracy for small models)
    algo_config = RTNWeightOnlyQuantConfig()

    # 3. Initialize the Quantizer
    print("Initializing MatMulNBitsQuantizer...")
    quantizer = MatMulNBitsQuantizer(
        model=model, 
        block_size=32,
        is_symmetric=True,
        algo_config=algo_config,
        accuracy_level=4
    )

    # 4. Process the model
    print("Processing quantization (this may take a minute)...")
    quantizer.process()

    # 5. Save the quantized model
    print(f"Saving to {output_model_path}...")
    # MatMulNBits typically doesn't need external data for smaller models, but let's be safe
    onnx.save_model(quantizer.model.model, output_model_path)

    if os.path.exists(output_model_path):
        size_int4 = os.path.getsize(output_model_path)
        print(f"\n✅ INT4 Quantization Complete!")
        print(f"INT4 Size: {size_int4 / 1024 / 1024:.1f} MB")
    else:
        print("Error: INT4 model file was not created.")

if __name__ == "__main__":
    quantize_setfit_int4()
