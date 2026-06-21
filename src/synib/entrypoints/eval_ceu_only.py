"""Load a saved best-model checkpoint and run val+test forward passes to
compute CEU metrics (using the CEU pickles wired in the method config).

We patch Agent.run to a no-op so only finalize() executes — which calls
mem_loader.load_best_model() and validator_tester.validate(test_set=True).
"""
import sys, os
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# IMPORTANT: patch Agent.run BEFORE running train's CLI, so only finalize() does work.
from synib.training.pipeline.agent import Agent
Agent.run = lambda self: None

# Run train's CLI (parses sys.argv) with the patched Agent.
from synib.entrypoints import train as _train_mod
_train_mod.cli()
