# Gearhead Quick Start Guide

Complete workflow from setup to using your trained model.

## Current Status

❌ **Model not trained yet** - You need to train it first!

## Step-by-Step Workflow

### Step 1: Prepare Data ⏳ START HERE

```bash
make prepare-data
```

Creates training data and tokenizer (~2 minutes)

### Step 2: Train Model 🚂 TAKES ~60-80 MIN

```bash
make train-small-rocm
```

Trains the model and saves to `outputs/small_model_rocm/final_model/`

### Step 3: Use Model 🎯 AFTER TRAINING

```bash
make infer  # Interactive mode
```

## What You Need to Do Now

```bash
# 1. Prepare data
make prepare-data

# 2. Train model (this takes time!)
make train-small-rocm

# 3. Then use it
make infer
```

## Full Guide

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete documentation.
