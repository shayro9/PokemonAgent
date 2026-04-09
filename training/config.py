LR = 3e-4
LR_DECAY = 0.9
N_STEPS = 4096
BATCH_SIZE = 256
GAMMA = 0.999
ENT_COEF = 0.1
LOG_FREQ = 500

# Shared architecture kwargs for AttentionPointerPolicy.
# Used by both the RL training pipeline and the supervised standalone runner
# so that a pretrained checkpoint is always compatible with RL fine-tuning.
POLICY_KWARGS = dict(
    context_hidden=256,
    move_hidden=128,
    trunk_hidden=256,
    n_attention_heads=4,
)