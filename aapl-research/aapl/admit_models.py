"""G2 correctness: method equivalence, causal inference and checkpoint recovery."""
import copy
from hashlib import sha256
import shutil
import time

import numpy as np
import torch

from .common import ROOT, digest, runtime, utc_now, write_json, read_json
from .data import Market
from .models import CausalEMA, chronos_pipeline, chronos_predict, chronos_training_batch, neural, quantile_point
from .training import Trainer, neural_predict


def test_ema():
    errors = []
    for device in ("cpu", "cuda"):
        for alpha in (0.0001, 0.01, 0.1, 0.7, 0.999, 1.0):
            for constant in (False, True):
                x = (torch.ones(2, 240, 3, dtype=torch.float64, device=device) if constant else
                     torch.randn(2, 240, 3, dtype=torch.float64, device=device)).requires_grad_()
                value = CausalEMA(alpha)(x)
                pieces = [x[:, 0]]
                for j in range(1, 240):
                    pieces.append(alpha * x[:, j] + (1 - alpha) * pieces[-1])
                expected = torch.stack(pieces, axis=1)
                torch.testing.assert_close(value, expected, rtol=1e-11, atol=1e-11)
                w = torch.randn_like(value)
                g1, = torch.autograd.grad((value * w).sum(), x, retain_graph=True)
                g2, = torch.autograd.grad((expected * w).sum(), x)
                torch.testing.assert_close(g1, g2, rtol=1e-10, atol=1e-10)
                errors.append(float((value - expected).abs().max().detach()))
    return {"cases": len(errors), "max_error_double": max(errors), "gradients_checked": True}


def recover_check(trainer, predict, output, rebuild=None):
    # Save mid-epoch, continue; restore the same point and compare the identical continuation.
    trainer.train_updates(2)
    trainer.save(output)
    losses1 = trainer.train_updates(5)
    pred1 = predict()
    state1 = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items() if v.is_floating_point()}
    trainer.restore(output)
    losses2 = trainer.train_updates(5)
    pred2 = predict()
    np.testing.assert_allclose(losses1, losses2, rtol=0, atol=0)
    np.testing.assert_allclose(pred1, pred2, rtol=0, atol=0)
    for name, v in trainer.model.state_dict().items():
        if name in state1:
            torch.testing.assert_close(v.cpu(), state1[name], rtol=0, atol=0)
    if rebuild is not None:
        fresh, fresh_predict = rebuild()
        fresh.restore(output)
        losses3 = fresh.train_updates(5)
        np.testing.assert_allclose(losses1, losses3, rtol=0, atol=0)
        np.testing.assert_allclose(pred1, fresh_predict(), rtol=0, atol=0)
        del fresh, fresh_predict
    # Wrong signature and corrupted bytes must fail before deserialization.
    original = trainer.signature
    trainer.signature = "deliberately_wrong"
    try:
        trainer.restore(output)
        raise AssertionError("Wrong identity accepted")
    except ValueError:
        pass
    trainer.signature = original
    corrupt = output.with_name(output.stem + "_corrupt.pt")
    corrupt.write_bytes(output.read_bytes()[:128])
    shutil.copyfile(output.with_suffix(".receipt.json"), corrupt.with_suffix(".receipt.json"))
    try:
        trainer.restore(corrupt)
        raise AssertionError("Corruption accepted")
    except ValueError:
        pass
    return {"mid_epoch_resume_identical_losses_weights_predictions": True,
            "fresh_model_optimizer_reconstruction_verified": rebuild is not None,
            "wrong_signature_rejected": True, "corrupted_bytes_rejected": True, "resume_updates": 5}


def main():
    device = runtime()
    started = time.perf_counter()
    output = ROOT / "outputs/model_admission"
    output.mkdir(parents=True, exist_ok=True)
    record = {"passed": False, "started_at_utc": utc_now(), "checks": {"ema": test_ema()}}
    write_json(ROOT / "outputs/model_admission.json", record)
    market = Market.load()
    days = sorted(set(market.dates))[:22]
    train, val = market.select(days[:20]), market.select(days[20:])
    scale = market.scale(days[:20])
    for name in ("xpatch", "patchtst"):
        runtime()
        config = dict(model=name, lookback=240, patch=16, stride=8, alpha=0.7, revin=True,
                      layers=3, d_model=64, dropout=0.2, loss="cumulative12_mse", lr=0.001,
                      weight_decay=0.01, batch_size=64)
        config.update(read_json(ROOT / 'outputs/optimization_admission.json')['adopted'].get(name, {}))
        model = neural(config).to(device)
        trainer = Trainer(model, config, market, train, scale, device, 2026, 27)
        fn = lambda: neural_predict(model, config, market, val[:64], scale, device)
        def rebuild():
            fresh_model = neural(config).to(device)
            fresh = Trainer(fresh_model, config, market, train, scale, device, 2026, 27)
            return fresh, lambda: neural_predict(fresh_model, config, market, val[:64], scale, device)
        checks = recover_check(trainer, fn, output / f"{name}.pt", rebuild)
        pred = fn()
        chunks = np.r_[neural_predict(model, config, market, val[:31], scale, device),
                       neural_predict(model, config, market, val[31:64], scale, device)]
        np.testing.assert_allclose(pred, chunks, rtol=2e-5, atol=1e-8)
        perm = np.arange(64)[::-1].copy()
        reordered = neural_predict(model, config, market, val[:64][perm], scale, device)
        np.testing.assert_allclose(pred, reordered[np.argsort(perm)], rtol=2e-5, atol=1e-8)
        checks["inference_batch_partition_invariant"] = True
        checks["inference_origin_order_invariant"] = True
        record["checks"][name] = checks
        write_json(ROOT / "outputs/model_admission.json", record)
        del trainer, model
        torch.cuda.empty_cache()
    runtime()
    pipe, receipt = chronos_pipeline(lora=True, rank=16, device=device)
    initial_trainable = {n: p.detach().cpu().clone() for n, p in pipe.model.named_parameters() if p.requires_grad}
    frozen_before = {n: sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                     for n, p in pipe.model.named_parameters() if not p.requires_grad}
    cfg = dict(model="chronos2_lora", lookback=240, lr=1e-5, batch_size=8,
               weight_decay=0.0, point="native_median", representation="minute_return")
    cfg.update(read_json(ROOT / 'outputs/optimization_admission.json')['adopted'].get('chronos2_lora', {}))
    trainer = Trainer(pipe.model, cfg, market, train, scale, device, 2026, 900, lora=True)
    fn = lambda: chronos_predict(pipe, market, val[:16], cfg, batch_size=16)
    def rebuild_lora():
        fresh_pipe, _ = chronos_pipeline(True, rank=16, device=device)
        fresh = Trainer(fresh_pipe.model, cfg, market, train, scale, device, 2026, 900, lora=True)
        return fresh, lambda: chronos_predict(fresh_pipe, market, val[:16], cfg, batch_size=16)
    checks = recover_check(trainer, fn, output / "chronos2_lora.pt", rebuild_lora)
    changed_names = [n for n, p in pipe.model.named_parameters() if p.requires_grad and
                     not torch.equal(p.detach().cpu(), initial_trainable[n])]
    assert changed_names, "No LoRA parameter actually changed"
    frozen_after = {n: sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                    for n, p in pipe.model.named_parameters() if not p.requires_grad}
    assert frozen_before == frozen_after, "Frozen pretrained base changed during LoRA"
    pred = fn()
    partitioned = chronos_predict(pipe, market, val[:16], cfg, batch_size=5)
    np.testing.assert_allclose(pred, partitioned, rtol=1e-4, atol=2e-7)
    isolated = chronos_predict(pipe, market, val[:1], cfg, batch_size=1)
    np.testing.assert_allclose(isolated, pred[:1], rtol=1e-4, atol=2e-7)
    order = np.arange(16)[::-1].copy()
    reordered = chronos_predict(pipe, market, val[:16][order], cfg, batch_size=5)
    np.testing.assert_allclose(pred, reordered[np.argsort(order)], rtol=1e-4, atol=2e-7)
    changed = copy.copy(market)
    changed.features = market.features.copy()
    changed.returns = market.returns.copy()
    changed.bars = market.bars.copy()
    origin = val[0]
    changed.features[origin + 1:] = 123.0
    changed.returns[origin + 1:] = 0.3
    changed.bars.iloc[origin + 1:, :4] *= 3.0
    np.testing.assert_array_equal(isolated, chronos_predict(pipe, changed, val[:1], cfg, batch_size=1))
    # Target labels may affect the loss, never the same forward pass's quantile predictions.
    batch = chronos_training_batch(market, val[:8], 240, device)
    pipe.model.eval()
    with torch.no_grad():
        a = pipe.model(**batch).quantile_preds
        batch["future_target"] = batch["future_target"] + 10
        b = pipe.model(**batch).quantile_preds
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    checks.update(causal_future_row_perturbation=True, label_changes_cannot_change_same_forward_predictions=True,
                  batch_partition_invariant=True, actual_lora_weights_updated=True, weight_identity=receipt)
    checks["origin_order_and_isolated_origin_invariance"] = True
    checks.update(changed_trainable_tensor_count=len(changed_names), all_frozen_base_tensors_unchanged=True)
    record["checks"]["chronos2_lora"] = checks
    # Point estimator integral is exact for a symmetric three-point quantile function.
    q = np.asarray([[[-1.0], [0.0], [1.0]]])
    np.testing.assert_allclose(quantile_point(q, [0.1, 0.5, 0.9], "quantile_integral_tailflat"), 0.0, atol=1e-15)
    record.update(passed=True, completed_at_utc=utc_now(), wall_seconds=time.perf_counter() - started,
                  scope="Core models; training-date small correctness cases only. No outer model scores computed.")
    write_json(ROOT / "outputs/model_admission.json", record)
    print({"passed": True, "wall_seconds": record["wall_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
