.PHONY: help install install-rocm test train prepare-data clean

help:
	@echo "Gearhead - Small Language Model for Equipment Diagnostics"
	@echo ""
	@echo "Available commands:"
	@echo "  make install           - Install dependencies (CUDA/CPU)"
	@echo "  make install-mps       - Install dependencies for Apple Silicon (M1/M2/M3)"
	@echo "  make install-rocm      - Install dependencies for ROCm (AMD GPUs)"
	@echo "  make prepare-data      - Prepare sample data and train tokenizer"
	@echo "  make test             - Run unit tests"
	@echo "  make test-rocm        - Test ROCm training setup"
	@echo "  make train-small      - Train small model (125M params)"
	@echo "  make train-small-rocm - Train small model with ROCm optimizations"
	@echo "  make train-small-mps   - Train small model on Apple Silicon"
	@echo "  make train-medium     - Train medium model (350M params)"
	@echo "  make generate-data     - Generate sample diagnostic data (1000 examples)"
	@echo "  make infer            - Run inference in interactive mode"
	@echo "  make infer-example    - Run example diagnosis"
	@echo "  make detect-gpu       - Detect available GPU and compute platform"
	@echo "  make clean            - Clean generated files"

install:
	pip install -r requirements.txt
	pip install -e .

install-rocm:
	@echo "Installing PyTorch with ROCm support..."
	@echo "Note: Make sure ROCm is installed on your system first"
	@echo "See: https://rocm.docs.amd.com/en/latest/"
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
	pip install -r requirements-rocm.txt
	pip install -e .

prepare-data:
	@echo "Preparing sample data..."
	python scripts/prepare_data.py --output data/processed
	@echo ""
	@echo "Creating sample text for tokenizer training..."
	mkdir -p data/text
	@echo "engine hydraulic fuel pressure temperature diagnostic error code symptom cause solution equipment excavator loader backhoe check inspect test verify replace repair" > data/text/sample.txt
	@echo ""
	@echo "Training tokenizer..."
	python scripts/prepare_data.py --train-tokenizer --text-files data/text/sample.txt --tokenizer-output tokenizer/tokenizer.json --vocab-size 32000

test:
	pytest tests/ -v

train-small:
	python scripts/train.py --config configs/small_config.yaml

train-small-rocm:
	@echo "Starting training with ROCm optimizations..."
	@chmod +x scripts/train_rocm.sh
	./scripts/train_rocm.sh --config configs/small_config_rocm.yaml

train-medium:
	python scripts/train.py --config configs/medium_config.yaml

detect-gpu:
	@echo "Detecting GPU and compute platform..."
	@python -c "import sys; sys.path.insert(0, 'src'); from gearhead.training.trainer import get_device_info; device_type, device_name, available = get_device_info(); print(f'Device Type: {device_type.upper()}'); print(f'Device Name: {device_name}'); print(f'Available: {available}')" 2>/dev/null || echo "Unable to detect GPU (PyTorch may not be installed)"

test-rocm:
	@echo "Testing ROCm training setup..."
	@chmod +x test_rocm_training.sh
	./test_rocm_training.sh

infer:
	@echo "Running inference in interactive mode..."
	@chmod +x scripts/infer_rocm.sh
	./scripts/infer_rocm.sh

infer-example:
	@echo "Running example diagnosis..."
	@chmod +x scripts/infer_rocm.sh
	./scripts/infer_rocm.sh --equipment "Caterpillar 320 Excavator" --symptom "Engine loses power under load" --error-codes P0087 SPN157

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache
	rm -rf outputs/
	rm -rf tokenizer/*.json

generate-data:
	@echo "Generating sample diagnostic data (1000 examples)..."
	python scripts/generate_sample_data.py
	@echo ""
	@echo "⚠️  This is synthetic test data for pipeline testing only!"
	@echo "For production, you need real diagnostic data."
	@echo "See DATA_REQUIREMENTS.md for details."

install-mps:
	@echo "Installing dependencies for Apple Silicon (MPS)..."
	pip install torch torchvision torchaudio
	pip install -r requirements.txt
	pip install -e .

train-small-mps:
	@echo "Starting training with Apple Silicon (MPS) optimizations..."
	python scripts/train.py --config configs/small_config_mps.yaml
