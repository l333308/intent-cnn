"""
ONNX 导出脚本
将训练好的 TextCNN 模型导出为 ONNX 格式，供 Go 推理服务使用
"""
import argparse
from pathlib import Path

import torch
import onnx

from model import TextCNN
from train import load_checkpoint


def export_onnx(
    checkpoint_path: str,
    output_path: str = None,
    max_len: int = 32,
    opset_version: int = 13,
    verify: bool = True,
):
    """
    导出模型为 ONNX 格式

    Args:
        checkpoint_path: 模型检查点路径
        output_path: ONNX 输出路径
        max_len: 最大序列长度
        opset_version: ONNX opset 版本
        verify: 是否验证导出的模型
    """
    checkpoint_path = Path(checkpoint_path)

    # 确定输出路径
    if output_path is None:
        output_path = checkpoint_path.parent / "textcnn.onnx"
    else:
        output_path = Path(output_path)

    print(f"Loading model from: {checkpoint_path}")
    model, vocab, label_encoder = load_checkpoint(checkpoint_path, device="cpu")
    model.eval()

    print(f"Vocab size: {len(vocab)}")
    print(f"Num classes: {label_encoder.num_classes}")

    # 创建 dummy 输入
    dummy_input = torch.randint(0, len(vocab), (1, max_len), dtype=torch.long)

    # 导出 ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print("ONNX export completed!")

    # 验证导出的模型
    if verify:
        print("Verifying ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")

        # 测试推理一致性
        try:
            import onnxruntime as ort

            print("Testing inference consistency...")

            # PyTorch 推理
            with torch.no_grad():
                pt_output = model(dummy_input).numpy()

            # ONNX Runtime 推理
            ort_session = ort.InferenceSession(str(output_path))
            ort_input = {"input_ids": dummy_input.numpy()}
            ort_output = ort_session.run(None, ort_input)[0]

            # 比较输出
            max_diff = abs(pt_output - ort_output).max()
            print(f"Max output difference: {max_diff:.6f}")

            if max_diff < 1e-5:
                print("Inference consistency check passed!")
            else:
                print("Warning: Output difference is larger than expected")

        except ImportError:
            print("onnxruntime not installed, skipping inference consistency check")

    # 打印模型信息
    print("\n" + "=" * 60)
    print("Export Summary:")
    print("=" * 60)
    print(f"  ONNX file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"  Max sequence length: {max_len}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Num classes: {label_encoder.num_classes}")
    print(f"  Intent labels: {list(label_encoder.label2id.keys())}")
    print("=" * 60)

    # 生成配置文件供 Go 服务使用
    config_path = output_path.parent / "model_config.json"
    import json
    config = {
        "model_path": output_path.name,
        "vocab_path": "vocab.json",
        "labels_path": "labels.json",
        "max_len": max_len,
        "num_classes": label_encoder.num_classes,
        "labels": label_encoder.label2id,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Config file saved to: {config_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export TextCNN to ONNX")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=32,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip model verification",
    )

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_len=args.max_len,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
