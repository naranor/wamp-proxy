import argparse
from pathlib import Path
from collections import OrderedDict
from transformers import AutoConfig
from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def export_model(model_id, output_dir, task="feature-extraction"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_quantized.onnx"

    print(f"--- Processing: {model_id} (Task: {task}) ---")

    config = AutoConfig.from_pretrained(model_id)

    # Determine model type for exporter
    model_type = config.model_type
    if "deberta-v3" in model_id.lower():
        model_type = "deberta-v2"

    print(f"Detected model type: {model_type}")

    # 1. Export to FP32 with attention outputs
    if not fp32_path.exists():
        print("Step 1: Exporting to FP32 (attentions enabled)...")

        onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            model_type=model_type, exporter="onnx", task=task
        )
        onnx_config = onnx_config_constructor(config)

        # Dynamic Custom Config to expose all attention layers
        class CustomOnnxConfig(onnx_config.__class__):
            @property
            def outputs(self) -> OrderedDict:
                # Get base outputs for the task (logits for classification, last_hidden_state for extraction)
                outputs = super().outputs
                # Append all attention layers
                layers = getattr(config, "num_hidden_layers", 12)
                for i in range(layers):
                    outputs[f"attentions.{i}"] = {
                        0: "batch",
                        1: "num_heads",
                        2: "sequence",
                        3: "sequence",
                    }
                return outputs

        model_kwargs = {"output_attentions": True}

        main_export(
            model_name_or_path=model_id,
            output=str(output_dir),
            task=task,
            custom_onnx_configs={"model": CustomOnnxConfig(config)},
            model_kwargs=model_kwargs,
            opset=14,
        )

    # 2. Quantization to INT8
    if fp32_path.exists() and not int8_path.exists():
        print("Step 2: Quantizing to INT8 (AVX-512 VNNI optimized)...")
        try:
            quantizer = ORTQuantizer.from_pretrained(output_dir, file_name="model.onnx")
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

            quantizer.quantize(
                save_dir=output_dir, quantization_config=qconfig, file_suffix="quantized"
            )
            print(f"✅ Successfully exported and quantized {model_id}")
        except Exception as e:
            print(f"❌ Quantization failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Transformer models to ONNX with Attention Weights"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--task",
        type=str,
        default="feature-extraction",
        help="Task type (feature-extraction or text-classification)",
    )

    args = parser.parse_args()
    export_model(args.model, args.out, args.task)
