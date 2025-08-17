from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

from app.config.config import settings
from app.config.logger import logger
from app.domain.models import ProgramPassingQuantile, AdmissionProbability
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.services.admission_monte_carlo import AdmissionMonteCarlo

_SOURCE_TZ = ZoneInfo("Europe/Moscow")  # сайт и расписание — МСК (tz-naive)


class RecalculateMonteCarloUseCase:
    """
    • Полностью очищает результаты Monte‑Carlo в БД
    • Запускает подсчёт, сохраняет новые probabilities / quantiles
    """

    def __init__(self, repo: ProgramRepository, n_simulations: int = 10_000):
        self._repo = repo
        self._n_sim = n_simulations

    @staticmethod
    def _df_from_records(recs):
        return pd.DataFrame([r.__dict__ for r in recs])

    def _run_model(self) -> AdmissionMonteCarlo:
        apps_df = self._df_from_records(self._repo.get_all_applications())
        appl_df = self._df_from_records(self._repo.get_all_applicants())
        stats_df = self._df_from_records(self._repo.get_all_submission_stats())

        # meta‑таблица всех направлений, нужна для exam_id
        prog_meta_df = self._repo.get_program_meta_df()

        # ← вычисляем истёкшие exam_id один раз до инициализации модели
        expired_exam_ids = self._expired_exam_ids(prog_meta_df)

        logger.info("Monte‑Carlo input: %d apps, %d applicants, %d stats",
                    len(apps_df), len(appl_df), len(stats_df))

        model = AdmissionMonteCarlo(
            applications=apps_df,
            applicants=appl_df,
            submission_stats=stats_df,
            programs_meta=prog_meta_df,
            n_simulations=self._n_sim,
            random_seed=None,
            expired_exam_ids=expired_exam_ids,
            freeze_expired_exams=settings.exam_freeze_enabled,
        )
        model.run_simulation()
        return model

    def execute(self) -> None:

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

        logger.info("→ Очистка старых Monte‑Carlo результатов…")
        self._repo.clear_admission_probabilities()
        self._repo.clear_program_quantiles()
        self._repo.commit()

        logger.info("→ Сохраняем: probabilities=%d, quantiles=%d",
                    len(prob_models), len(quant_models))

        self._repo.add_program_quantiles_bulk(quant_models)
        self._repo.add_admission_probabilities_bulk(prob_models)
        self._repo.commit()

        logger.info("Monte‑Carlo результаты обновлены в БД.")

    def _expired_exam_ids(self, prog_meta_df: pd.DataFrame) -> set[str]:
        """
        Считает множество exam_id, для которых последнее ВИ прошло более, чем на settings.exam_grace_hours.
        exam_id = department_code (обычные) или department_code + "__eng" (международные).
        """
        sessions = self._repo.get_all_exam_sessions()
        if not sessions:
            logger.info("Расписание экзаменов пустое — freeze по экзаменам не применяем.")
            return set()

        df = pd.DataFrame([{"program_code": s.program_code, "dt": s.dt} for s in sessions])
        df = df.merge(prog_meta_df, on="program_code", how="left")
        if df.empty:
            return set()

        df["exam_id"] = df.apply(
            lambda r: f"{r['department_code']}__eng" if bool(r["is_international"]) else str(r["department_code"]),
            axis=1
        )
        last_dt = df.groupby("exam_id")["dt"].max()

        # now в МСК, сравнение с tz-naive датами: берём now_naive в МСК
        now_msk_naive = datetime.now(settings.timezone).astimezone(_SOURCE_TZ).replace(tzinfo=None)
        grace = timedelta(hours=int(settings.exam_grace_hours))

        expired = set()
        for eid, dt in last_dt.items():
            if pd.isna(dt):
                continue
            if now_msk_naive >= (dt + grace):
                expired.add(eid)

        # Логи
        total_e = len(last_dt)
        exp_e = len(expired)
        if total_e:
            min_last, max_last = last_dt.min(), last_dt.max()
        else:
            min_last = max_last = None
        logger.info(
            "Экзамены: всего exam_id=%d; истёкших=%d; интервал последних дат: [%s … %s]; grace=%dh",
            total_e, exp_e,
            (min_last.strftime('%d.%m %H:%M') if min_last else "—"),
            (max_last.strftime('%d.%m %H:%M') if max_last else "—"),
            int(settings.exam_grace_hours),
        )
        if exp_e:
            sample = list(sorted(expired))[:8]
            logger.debug("Примеры истёкших exam_id: %s", ", ".join(sample))
        return expired
