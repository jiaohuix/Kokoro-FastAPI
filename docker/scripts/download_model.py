#!/usr/bin/env python3
"""Download and prepare Kokoro v1.0 model from ModelScope (jiaohui/kokoro)."""

import os
import shutil
from pathlib import Path
import argparse
from modelscope import snapshot_download
from loguru import logger


def download_model(output_dir: str) -> None:
    """Download model from ModelScope and copy to output directory."""
    try:
        model_id = 'jiaohui/kokoro'
        model_file_name = "kokoro-v1_0.pth"
        config_file_name = "config.json"  # 如果下载包里有这个文件

        # 在容器中固定缓存目录为 /app/models（WORKDIR=/app）
        cache_dir = Path("/app/models")
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading model '{model_id}' from ModelScope to {cache_dir}")
        model_dir = snapshot_download(model_id, cache_dir=str(cache_dir))
        logger.info(f"Model downloaded to: {model_dir}")

        # 递归查找 kokoro-v1_0.pth
        source_pth = None
        source_config = None
        for root, dirs, files in os.walk(model_dir):
            if model_file_name in files:
                source_pth = Path(root) / model_file_name
            if config_file_name in files:
                source_config = Path(root) / config_file_name

        if source_pth is None:
            raise FileNotFoundError(f"{model_file_name} not found in downloaded directory: {model_dir}")

        # 目标目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 复制主模型文件
        dest_pth = output_path / model_file_name
        if dest_pth.exists():
            file_size_mb = dest_pth.stat().st_size / (1024 * 1024)
            if file_size_mb > 300:  # 假设正常文件 > 300MB
                logger.info(f"Model file already exists and size looks correct ({file_size_mb:.1f} MB): {dest_pth}")
            else:
                logger.warning(f"Existing file size too small ({file_size_mb:.1f} MB), re-copying...")
                shutil.copy2(source_pth, dest_pth)
                logger.info(f"Model file re-copied to: {dest_pth}")
        else:
            shutil.copy2(source_pth, dest_pth)
            logger.info(f"Model file copied to: {dest_pth}")

        # 如果有 config.json，也复制
        if source_config:
            dest_config = output_path / config_file_name
            shutil.copy2(source_config, dest_config)
            logger.info(f"Config file copied to: {dest_config}")
        else:
            logger.warning("config.json not found in ModelScope download. "
                           "The project may use default config or need manual addition.")

        logger.success(f"✓ Model preparation completed in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download or prepare model: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download Kokoro v1.0 model from ModelScope")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for model files (e.g. api/src/models/v1_0)"
    )
    args = parser.parse_args()
    download_model(args.output)


if __name__ == "__main__":
    main()