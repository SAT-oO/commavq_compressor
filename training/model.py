"""
NextFramePredictor: Predicts the probability distribution over 1024 VQ tokens
for each of 128 spatial positions in the next frame, given T=8 previous frames.

Architecture: TransformerEncoder on T*128 context tokens, using the last-frame
output positions to produce 128 × 1024 logits for the next frame.

~4.5M parameters → ~9 MB float16 / ~4.5 MB int8.
"""

import numpy as np
import torch
import torch.nn as nn

VOCAB_SIZE = 1024
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
FFN_DIM = 768        # wider FFN for ~4.5M total params
CONTEXT_FRAMES = 8
TOKENS_PER_FRAME = 128   # 8×16 spatial grid per frame
FRAME_ROWS = 8
FRAME_COLS = 16


class NextFramePredictor(nn.Module):
    """
    Transformer encoder that maps T context frames → next-frame token distributions.

    Input  x: (B, T, 128)  int64 token IDs
    Output  : (B, 128, 1024) float32 logits
    """

    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # Decompose 8×16 spatial position into row + col embeddings (saves params,
        # explicitly models the 2-D grid structure).
        self.row_embed = nn.Embedding(FRAME_ROWS, D_MODEL)
        self.col_embed = nn.Embedding(FRAME_COLS, D_MODEL)

        self.temporal_embed = nn.Embedding(CONTEXT_FRAMES, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=0.0,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
        )
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Pre-compute spatial position indices (row, col) for each of 128 positions.
        row_idx = torch.arange(FRAME_ROWS).repeat_interleave(FRAME_COLS)  # (128,)
        col_idx = torch.arange(FRAME_COLS).repeat(FRAME_ROWS)             # (128,)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, S) int64 token IDs; T=CONTEXT_FRAMES, S=TOKENS_PER_FRAME
        Returns:
            logits: (B, S, VOCAB_SIZE) float32
        """
        B, T, S = x.shape

        tok_emb  = self.token_embed(x)                                    # (B, T, S, D)
        sp_emb   = self.row_embed(self.row_idx) + self.col_embed(self.col_idx)  # (S, D)
        tp_emb   = self.temporal_embed(torch.arange(T, device=x.device)) # (T, D)

        # Broadcast and sum positional embeddings.
        x_emb = (tok_emb
                 + sp_emb[None, None]               # (1, 1, S, D)
                 + tp_emb[None, :, None])            # (1, T, 1, D)

        x_flat = x_emb.reshape(B, T * S, D_MODEL)   # (B, T*S, D)
        out    = self.transformer(x_flat)             # (B, T*S, D)
        out_last = out[:, -S:]                        # (B, S, D)  ← most-recent frame
        return self.output_head(out_last)             # (B, S, VOCAB_SIZE)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Context-building helpers (shared by compress / decompress / training)

def build_context(tokens: np.ndarray, t: int, T: int = CONTEXT_FRAMES) -> np.ndarray:
    """
    Return T context frames for predicting frame t.

    tokens : (num_frames, TOKENS_PER_FRAME) int16/int64
    Returns: (T, TOKENS_PER_FRAME) same dtype
             Left-padded by repeating frame 0 when t < T.
    """
    start = max(0, t - T)
    avail = tokens[start:t]           # up to T frames
    n     = len(avail)
    if n == T:
        return avail
    # Left-pad with the earliest available frame (frame 0 if t > 0, else zeros).
    if t > 0:
        pad_frame = tokens[0:1]
    else:
        pad_frame = np.zeros((1, tokens.shape[1]), dtype=tokens.dtype)
    pad = np.repeat(pad_frame, T - n, axis=0)
    return np.concatenate([pad, avail], axis=0)


def build_context_batch(
    tokens_batch: np.ndarray, t: int, T: int = CONTEXT_FRAMES
) -> np.ndarray:
    """
    Vectorised version for a batch of token sequences.

    tokens_batch: (B, num_frames, TOKENS_PER_FRAME)
    Returns     : (B, T, TOKENS_PER_FRAME)
    """
    start = max(0, t - T)
    avail = tokens_batch[:, start:t, :]   # (B, n, S)
    n     = avail.shape[1]
    if n == T:
        return avail.copy()
    # Left-pad with frame 0 repeated.
    pad = np.repeat(tokens_batch[:, 0:1, :], T - n, axis=1)   # (B, T-n, S)
    if n > 0:
        return np.concatenate([pad, avail], axis=1)
    return pad


# Model I/O

def load_model(path: str, device: str = "cpu") -> NextFramePredictor:
    """Load a saved model (supports float32 and float16 state dicts).

    For float32 weights: loaded as-is.
    For float16 weights: converted back to float32 for inference.
    """
    model = NextFramePredictor().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_model_quantised(path, device: str = "cpu") -> NextFramePredictor:
    """Load a float16 checkpoint and reconstruct to float32.

    Crucially, the float16 → float32 conversion is deterministic, so the
    model produces IDENTICAL probabilities to the encoder that also loaded the
    same float16 checkpoint.  Always use this when the weights on disk are f16.
    """
    return load_model(path, device=device)


def save_model_f16(model: NextFramePredictor, path) -> None:
    """Save model weights as float16 (half-precision) to reduce file size.

    path: str/Path for a file, or a writable file-like object (e.g. io.BytesIO).

    IMPORTANT: after calling this, recreate the model from the saved file so
    that BOTH encoder and decoder operate on identically quantised weights
    (float32 → float16 → float32).  See `rebuild_from_f16`.
    """
    state_f16 = {
        k: v.half() if v.is_floating_point() else v
        for k, v in model.state_dict().items()
    }
    torch.save(state_f16, path)


def rebuild_from_f16(buf_or_path, device: str = "cpu") -> NextFramePredictor:
    """Round-trip a model through float16 quantisation.

    Equivalent to: save_model_f16(model, buf); load_model(buf, device).
    Returns a NEW model with weights that are identical to what decompress.py
    will load from the submission zip.
    """
    from pathlib import Path as _Path
    if isinstance(buf_or_path, (str, _Path)):
        return load_model(buf_or_path, device=device)
    # buf_or_path is a BytesIO: rewind to read
    buf_or_path.seek(0)
    return load_model_quantised(buf_or_path, device=device)
