# 🚀 Training on Google Colab

## ⚡ Quick Start (3 Steps)

1. Go to: https://colab.research.google.com/
2. **File → Upload notebook** → Select `PokemonAgent_Colab_Training.ipynb`
3. Update **Cell 2** with your repo URL, then run **Cells 1-7** in order

**That's it!** ✨

---

## 🎯 Cell-by-Cell Overview

| Cell | Task | Time |
|------|------|------|
| 1 | Install Node.js | 30s |
| 2 | Clone repo + Python deps | 2 min |
| 3 | Build Pokémon Showdown | 3 min |
| 4 | Start server | 5s |
| 5 | Verify server | 30s |
| 6 | Configure W&B (optional) | 5s |
| **7** | **🎯 RUN TRAINING** | **5 min - 5 hrs** |
| 8 | Download model | 1 min |

---

## 📝 Customize Cell 7

### Quick Test (5 min)
```bash
python -m training.train --train-team steelix --pool toxapex --timesteps 10000 --rounds-per-opponent 100 --device cuda
```

### Balanced (1-2 hrs)
```bash
python -m training.train --train-team garchomp --pool-all --timesteps 200000 --rounds-per-opponent 2000 --device cuda
```

### Full (3-5 hrs)
```bash
python -m training.train --train-team steelix --pool-all --timesteps 500000 --rounds-per-opponent 2000 --eval-every-timesteps 50000 --eval-episodes 100 --device cuda
```

### All Available Options
```bash
python -m training.train --help
```

---

## ⚙️ Key Settings

| Setting | What to Do |
|---------|-----------|
| **Cell 2** | ⚠️ Change `repo_url` to YOUR GitHub repository |
| **Cell 6** | (Optional) Add W&B API key from https://wandb.ai/settings/api |
| **Cell 7** | Customize training parameters above |

---

## ⚠️ Important Notes

| Issue | Solution |
|-------|----------|
| Server won't connect | Wait 10s longer in Cell 5, re-run |
| Out of memory | Reduce `--timesteps` or `--rounds-per-opponent` |
| No GPU available | Runtime → Change runtime type → GPU |
| Session timeout (12 hrs) | Download model with Cell 8 before timeout |

---

## 📊 What Happens

```
Colab sandboxed environment
  ↓
Node.js installed
  ↓
Repository cloned (your code)
  ↓
Pokémon Showdown server starts (localhost:8000)
  ↓
🎯 Training begins (GPU-accelerated)
  ↓
Model saved to /content/PokemonAgent/data/1v1/
  ↓
Downloaded to your computer (Cell 8)
```

---

## 📊 Expected Times

- **Quick test**: 5 min (10k timesteps)
- **Short**: 30 min (50k timesteps)  
- **Medium**: 1-2 hrs (200k timesteps)
- **Long**: 3-5 hrs (500k timesteps)

---

## 🎬 After Training

1. **Cell 8**: Download your trained model (auto-saves as `.zip`)
2. **W&B**: Check metrics at https://wandb.ai/your-username/pokemon-rl
3. **Local**: Extract model, use for battles, or fine-tune with more training

---

## 💡 Pro Tips

- Use different `--seed` values for ensemble training
- Monitor W&B dashboard in real-time during training
- Save notebook frequently (Colab auto-saves too)
- Use `--eval-every-timesteps` for periodic evaluation

---

## 📞 Troubleshooting

**"Cannot reach server"** → Cell 5 will auto-retry. If it fails, re-run Cell 4.

**"ImportError" in training** → Run Cell 2 again to install dependencies.

**"Out of memory"** → Reduce timesteps: `--timesteps 100000` instead of `500000`

**"No GPU"** → Runtime → Change runtime type → GPU tab → Select GPU

---

**Ready?** Go to https://colab.research.google.com/ and upload the notebook! 🎮
