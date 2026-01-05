# RunPod Deployment Checklist

Quick checklist for deploying to RunPod. Use this to ensure you don't miss any steps.

---

## ✅ Pre-Deployment (Local Machine)

- [ ] **Push code to GitHub**
  ```bash
  cd /path/to/llm-golf
  git add -A
  git commit -m "Ready for RunPod deployment"
  git push origin main
  ```

- [ ] **Get HuggingFace token**
  - Go to: https://huggingface.co/settings/tokens
  - Copy token (starts with `hf_`)
  - Save somewhere secure

- [ ] **Get ClearML credentials (optional)**
  - Go to: https://app.clear.ml
  - Settings → Workspace → Create credentials
  - Copy API key, secret, and host URL

---

## ✅ RunPod Setup

- [ ] **Create RunPod account**
  - Go to: https://runpod.io
  - Add payment method
  - Add $20-50 credits

- [ ] **Launch pod**
  - Template: **RunPod Pytorch 2.4.0**
  - GPU: **RTX 4090** (recommended)
  - Container Disk: **30 GB**
  - Volume Disk: **50 GB**
  - Volume Mount: **/workspace**
  - Enable **SSH**

- [ ] **Get SSH credentials**
  - Copy SSH command from RunPod console
  - Example: `ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519`

---

## ✅ Pod Configuration

- [ ] **Connect via SSH**
  ```bash
  ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519
  ```

- [ ] **Set environment variables**
  ```bash
  export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'
  export CLEARML_API_ACCESS_KEY='your_key'
  export CLEARML_API_SECRET_KEY='your_secret'
  export CLEARML_API_HOST='https://api.clear.ml'
  ```

- [ ] **Run setup script**
  ```bash
  cd /workspace
  git clone https://github.com/Tumph/Psuedoptimal.git
  cd Psuedoptimal
  bash runpod/setup.sh
  ```

- [ ] **Verify GPU**
  ```bash
  nvidia-smi
  # Should show RTX 4090 with ~24GB VRAM
  ```

---

## ✅ Training Execution

- [ ] **Start training in background**
  ```bash
  cd /workspace/Psuedoptimal
  bash runpod/run_training_background.sh
  ```

- [ ] **Verify training started**
  ```bash
  # Should see PID and log file path
  ps aux | grep train_dsl
  ```

- [ ] **Monitor logs**
  ```bash
  tail -f logs/training_*.log
  ```

- [ ] **Check ClearML dashboard**
  - Go to: https://app.clear.ml
  - Navigate to: Projects → LLM-DSL → GRPO-Training
  - Verify metrics are logging

---

## ✅ During Training (Optional)

- [ ] **Monitor GPU usage**
  ```bash
  watch -n 1 nvidia-smi
  # Should show ~20-22GB VRAM used
  ```

- [ ] **Check training progress**
  ```bash
  grep "Step" logs/training_*.log | tail -20
  ```

- [ ] **View encoding examples**
  ```bash
  grep -A 10 "ENCODING VISUALIZATION" logs/training_*.log
  ```

---

## ✅ After Training

- [ ] **Verify training completed**
  ```bash
  tail -50 logs/training_*.log
  # Should see "=== Training Complete ==="
  ```

- [ ] **Check model was saved**
  ```bash
  ls -lh outputs/llm-dsl/final/
  # Should see adapter_model.safetensors (~35MB)
  ```

- [ ] **Download trained model** (from local machine)
  ```bash
  scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/outputs/llm-dsl/final ./trained_model_3b
  ```

- [ ] **Download logs** (from local machine)
  ```bash
  scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/logs ./training_logs
  ```

- [ ] **Export ClearML data**
  - Go to ClearML experiment page
  - Click "Scalars" → "Export to CSV"
  - Download encoding examples from "Debug Samples"

- [ ] **STOP THE POD** ⚠️
  - Go to RunPod console
  - Click "Stop" or "Terminate"
  - Verify billing stopped

---

## ✅ Analysis

- [ ] **Review training curves**
  - Check loss progression (should decrease)
  - Check reward progression (should increase)
  - Identify convergence point

- [ ] **Analyze compression evolution**
  ```bash
  grep "ENCODING" training_logs/*.log | less
  ```
  - Compare early vs late encodings
  - Measure token reduction

- [ ] **Test trained model**
  - Use test script from RUNPOD_GUIDE.md
  - Generate encodings for new problems
  - Verify Generator can decode them

- [ ] **Calculate final metrics**
  - Average compression ratio
  - Test pass rate
  - Training cost (hours × $/hour)

---

## Expected Results

After ~3.5 hours of training:

**Costs:**
- RTX 4090: ~$1.54 total

**Model:**
- Trained LoRA adapter: ~35 MB
- Can compress Python → compact encodings
- Generator can decode encodings → working Python

**Next Steps:**
- Fine-tune further if needed
- Test on new problems
- Share results with community

---

## Quick Commands Reference

```bash
# Start training
bash runpod/run_training_background.sh

# Monitor
tail -f logs/training_*.log
watch -n 1 nvidia-smi

# Check if running
ps aux | grep train_dsl

# Download results (local machine)
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/outputs ./
scp -P XXXXX -r root@X.X.X.X:/workspace/Psuedoptimal/logs ./
```

---

**Ready? Let's train! 🚀**
