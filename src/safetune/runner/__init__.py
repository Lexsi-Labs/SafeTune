"""safetune.runner — high-level pillar API.

Usage::

    from safetune.runner import harden, steer, recover, unlearn

    # Harden
    trainer = harden.LisaTrainer(model, tok, lisa_rho=0.1)
    path = trainer.train(train_ds, "/tmp/lisa")
    metrics = trainer.eval("lisa_run", path)

    # Recover
    trainer = recover.CThetaTrainer(model, base_model=base, aligned_model=aligned)
    patched = trainer.apply()

    # Steer
    trainer = steer.RefusalDirectionTrainer(model, tok)
    wrapped = trainer.calibrate(harmful=harmful, harmless=harmless)

    # Unlearn
    trainer = unlearn.RMUTrainer(model, layer_id=7)
    clean  = trainer.unlearn(forget=forget_ds, retain=retain_ds)

Results are written to safetune/src/safetune/results/ by default.
Override with ``results_dir`` on any trainer.
"""

from safetune.runner import harden, steer, recover, unlearn
from safetune.runner.harden import load_harden_data
from safetune.runner.steer import load_steer_data
from safetune.runner.recover import load_recover_data
from safetune.runner.unlearn import load_unlearn_data

__all__ = [
    "harden", "steer", "recover", "unlearn",
    "load_harden_data", "load_steer_data", "load_recover_data", "load_unlearn_data",
]
