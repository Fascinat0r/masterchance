from typing import List

import pandas as pd

from app.config.logger import logger
from app.domain.models import ProgramPassingQuantile, AdmissionProbability
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.services.admission_monte_carlo import AdmissionMonteCarlo


class RecalculateMonteCarloUseCase:
    """
    • Полностью очищает результаты Monte‑Carlo в БД
    • Запускает подсчёт, сохраняет новые probabilities / quantiles
    """

    def __init__(self, repo: ProgramRepository, n_simulations: int = 10_000):
        self._repo = repo
        self._n_sim = n_simulations

    def _df_from_records(self, recs):
        return pd.DataFrame([r.__dict__ for r in recs])

    def _run_model(self) -> AdmissionMonteCarlo:
        apps_df = self._df_from_records(self._repo.get_all_applications())
        appl_df = self._df_from_records(self._repo.get_all_applicants())
        stats_df = self._df_from_records(self._repo.get_all_submission_stats())

        logger.info("Monte‑Carlo input: %d apps, %d applicants, %d stats",
                    len(apps_df), len(appl_df), len(stats_df))

        model = AdmissionMonteCarlo(
            applications=apps_df,
            applicants=appl_df,
            submission_stats=stats_df,
            n_simulations=self._n_sim,
            random_seed=None,
        )
        model.run_simulation()
        return model

    def execute(self) -> None:
        logger.info("→ Очистка старых Monte‑Carlo результатов…")
        self._repo.clear_admission_probabilities()
        self._repo.clear_program_quantiles()
        self._repo.commit()

        logger.info("→ Запуск Monte‑Carlo…")
        monte = self._run_model()

        # --- преобразуем результаты в доменные модели ------------------------
        quant_models: List[ProgramPassingQuantile] = [
            ProgramPassingQuantile(program_code=code, **vals)
            for code, vals in monte.get_passing_score_quantiles().items()
        ]

        prob_models: List[AdmissionProbability] = []
        for aid, mapping in monte.get_probabilities().items():
            prob_models.extend([
                AdmissionProbability(applicant_id=aid,
                                     program_code=prog,
                                     probability=prob)
                for prog, prob in mapping.items()
            ])

        logger.info("→ Сохраняем: probabilities=%d, quantiles=%d",
                    len(prob_models), len(quant_models))

        self._repo.add_program_quantiles_bulk(quant_models)
        self._repo.add_admission_probabilities_bulk(prob_models)
        self._repo.commit()

        logger.info("Monte‑Carlo результаты обновлены в БД.")
