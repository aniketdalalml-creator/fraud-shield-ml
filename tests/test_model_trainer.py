from __future__ import annotations

import numpy as np

from src.train import ModelTrainer


def test_model_trainer_fit_small_dataset() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 6))
    y = np.concatenate([np.zeros(100, dtype=int), np.ones(20, dtype=int)])
    rng.shuffle(y)

    trainer = ModelTrainer(random_state=0, search_n_iter=3, cv_folds=3)
    trainer.fit(X, y)

    assert trainer.best_estimator_ is not None
    assert trainer.best_params_ is not None
    assert trainer.best_cv_score_ is not None
    assert trainer.training_time_s_ is not None
    assert trainer.best_estimator_.predict(X).shape == (len(X),)
