# app/services/admission_monte_carlo.py
from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import gaussian_kde

from app.config.config import settings
from app.config.logger import logger

MAX_EXAM_SCORE = 100
MAX_ID_ACHIEVEMENTS = 10
MAX_TOTAL_SCORE = MAX_EXAM_SCORE + MAX_ID_ACHIEVEMENTS
SCORE_COL = "vi_score"

KDE_RESAMPLE_N = 10_000
RANK_SCALE = 100


def _kde_resample(kde: gaussian_kde, n: int, rng: np.random.Generator) -> np.ndarray:
    """Безопасная выборка из gaussian_kde для случаев, когда random_state недоступен."""
    try:
        return kde.resample(n, random_state=rng).ravel()
    except TypeError:
        seed = int(rng.integers(0, 2 ** 32 - 1, dtype=np.uint32))
        saved_state = np.random.get_state()
        try:
            np.random.seed(seed)
            return kde.resample(n).ravel()
        finally:
            np.random.set_state(saved_state)


@njit(cache=True)
def _simulate_admission_numba(priority, program_idx, applicant_idx,
                              total_score, seats_init, jitter,
                              max_priority) -> Tuple[np.ndarray, np.ndarray]:
    """
    Быстрая симуляция распределения мест:
      • по приоритетам 1..max_priority;
      • 'jitter' ломает тай-брейки.
    Возвращает:
      admitted[A] = p_idx или -1
      passing[P]  = худший (минимальный) принятый балл или -1, если мест нет.
    """
    A = applicant_idx.max() + 1
    P = seats_init.size
    N = priority.size

    admitted = np.full(A, -1, np.int32)
    passing = np.full(P, -1, np.int16)

    max_seats = np.max(seats_init)
    seat_cnt = np.zeros(P, np.int32)
    seats_left = seats_init.copy()

    tab_app = np.full((P, max_seats), -1, np.int32)
    tab_score = np.full((P, max_seats), -1, np.int16)
    tab_rank = np.full((P, max_seats), 0, np.int32)

    worst_score = np.full(P, -1, np.int16)
    worst_rank = np.full(P, -1, np.int32)
    worst_slot = np.zeros(P, np.int32)

    for pr in range(1, max_priority + 1):
        for i in range(N):
            if priority[i] != pr:
                continue
            appl = applicant_idx[i]
            if admitted[appl] != -1:
                continue

            prog = program_idx[i]
            raw = np.int16(total_score[i])
            rank = raw * RANK_SCALE + np.int32(jitter[i] * RANK_SCALE)

            if seats_left[prog] > 0:
                s = seat_cnt[prog]
                tab_app[prog, s] = appl
                tab_score[prog, s] = raw
                tab_rank[prog, s] = rank

                seat_cnt[prog] += 1
                seats_left[prog] -= 1
                admitted[appl] = prog

                if worst_rank[prog] == -1 or rank < worst_rank[prog]:
                    worst_rank[prog] = rank
                    worst_score[prog] = raw
                    worst_slot[prog] = s
                continue

            if rank > worst_rank[prog]:
                kick = worst_slot[prog]
                old_appl = tab_app[prog, kick]
                admitted[old_appl] = -1

                tab_app[prog, kick] = appl
                tab_score[prog, kick] = raw
                tab_rank[prog, kick] = rank
                admitted[appl] = prog

                wr, ws, wslt = rank, raw, kick
                for t in range(seat_cnt[prog]):
                    r = tab_rank[prog, t]
                    if r < wr:
                        wr, ws, wslt = r, tab_score[prog, t], t
                worst_rank[prog] = wr
                worst_score[prog] = ws
                worst_slot[prog] = wslt

    for p in range(P):
        if seat_cnt[p] > 0:
            passing[p] = worst_score[p]

    return admitted, passing


class AdmissionMonteCarlo:
    """
    MC-модель шансов зачисления (без opt-out), с поддержкой «заморозки» нулей по истёкшим экзаменам.
      • Импутация ВИ: по personal μ → CDF конкретного экзамена → глобальная CDF.
      • Freeze: если для (applicant×exam_id) нет ни одной известной оценки и exam_id помечен как истёкший,
        нули остаются нулями (импутации нет).
    """

    def __init__(self,
                 applications: pd.DataFrame,
                 applicants: pd.DataFrame | None,
                 submission_stats: pd.DataFrame,
                 programs_meta: pd.DataFrame,
                 *,
                 n_simulations: int = 10_000,
                 random_seed: int | None = None,
                 expired_exam_ids: set[str] | None = None,
                 freeze_expired_exams: bool | None = None):
        self.n_sim = n_simulations
        self.rng = np.random.default_rng(random_seed)

        # Freeze экзаменов
        self.freeze_expired_exams = (
            settings.exam_freeze_enabled if freeze_expired_exams is None else bool(freeze_expired_exams)
        )
        self._expired_exam_ids_in = set(expired_exam_ids or [])
        logger.info(
            "AdmissionMonteCarlo: подготовка данных… (exam-freeze: %s)",
            "ON" if self.freeze_expired_exams else "OFF",
        )

        self._rows = len(applications)

        # --- Индексация ------------------------------------------------------
        self._applicant2idx = {aid: i for i, aid in enumerate(applications["applicant_id"].unique())}
        self._program2idx = {c: i for i, c in enumerate(applications["program_code"].unique())}
        self.n_applicants = len(self._applicant2idx)
        self.n_programs = len(self._program2idx)

        # exam_id: department_code (обычные) или department_code__eng (международные)
        meta = programs_meta.set_index("program_code")
        self.exam_id = np.empty(self._rows, dtype="U24")
        for i, p_code in enumerate(applications["program_code"]):
            row = meta.loc[p_code]
            dept = str(row["department_code"])
            self.exam_id[i] = f"{dept}__eng" if bool(row["is_international"]) else dept

        self._exam2idx = {eid: j for j, eid in enumerate(np.unique(self.exam_id))}
        self.exam_idx = np.vectorize(self._exam2idx.get)(self.exam_id).astype(np.int32)
        self.n_exams = len(self._exam2idx)
        logger.info("   найдено %d различных экзаменов.", self.n_exams)

        # Какие exam_id истёкли (по индексу экзамена)
        self.expired_exam_mask = np.zeros(self.n_exams, dtype=bool)
        if self._expired_exam_ids_in:
            for eid, j in self._exam2idx.items():
                if eid in self._expired_exam_ids_in:
                    self.expired_exam_mask[j] = True
        n_expired_ids = int(self.expired_exam_mask.sum())

        # --- Вектора заявок --------------------------------------------------
        self.applicant_idx = applications["applicant_id"].map(self._applicant2idx).to_numpy(np.int32, copy=False)
        self.program_idx = applications["program_code"].map(self._program2idx).to_numpy(np.int32, copy=False)
        self.priority = applications["priority"].to_numpy(np.int16, copy=False)
        # Нормализация приоритетов: pr<=0 → глобальный max+1 (считаем наихудшим)
        if (self.priority <= 0).any():
            pr = self.priority.copy()
            pos = pr[pr > 0]
            max_pos = int(pos.max()) if pos.size else 1
            pr[pr <= 0] = max_pos + 1
            fixed = int((self.priority <= 0).sum())
            self.priority = pr
            logger.info("Нормализованы приоритеты: исправлено %d записей с pr<=0 → %d",
                        fixed, max_pos + 1)

        self.vi_score = applications[SCORE_COL].to_numpy(np.int16, copy=False)
        self.id_ach = applications["id_achievements"].to_numpy(np.int16, copy=False)

        # Ряды по (applicant, exam) и по applicant
        self._rows_by_app_exam: Dict[Tuple[int, int], np.ndarray] = {}
        for r, (a, e) in enumerate(zip(self.applicant_idx, self.exam_idx)):
            self._rows_by_app_exam.setdefault((a, e), []).append(r)
        for k in list(self._rows_by_app_exam):
            self._rows_by_app_exam[k] = np.asarray(self._rows_by_app_exam[k], dtype=np.int32)

        # --- Персональные средние (μ) ---------------------------------------
        logger.info("→ вычисляем personal_mu …")
        known = (self.vi_score > 0) & (self.vi_score < MAX_EXAM_SCORE)
        sums = np.bincount(self.applicant_idx[known],
                           self.vi_score[known].astype(np.float32),
                           minlength=self.n_applicants)
        cnts = np.bincount(self.applicant_idx[known], minlength=self.n_applicants)
        self.personal_mu = np.zeros(self.n_applicants, np.float32)
        mask = cnts > 0
        self.personal_mu[mask] = sums[mask] / cnts[mask]
        logger.debug("   персональный μ есть у %d / %d абитуриентов.", mask.sum(), self.n_applicants)

        # --- KDE → CDF по каждому экзамену (fallback → глобальная CDF) ------
        logger.info("→ строим KDE → CDF (%d точек)…", KDE_RESAMPLE_N)
        self._exam_cdf = np.zeros((self.n_exams, MAX_EXAM_SCORE), np.float32)

        global_samples = self.vi_score[known].astype(np.float64)
        g_kde = gaussian_kde(global_samples, bw_method="scott")
        g_raw = _kde_resample(g_kde, KDE_RESAMPLE_N, self.rng)
        g_raw = np.clip(np.rint(g_raw), 1, MAX_EXAM_SCORE).astype(int)
        g_hist = np.bincount(g_raw, minlength=MAX_EXAM_SCORE + 1)[1:].astype(np.float32)
        self.global_cdf = np.cumsum(g_hist / g_hist.sum())

        for eid, j in self._exam2idx.items():
            mask_e = (self.exam_idx == j) & known
            samples = self.vi_score[mask_e].astype(np.float64)
            if samples.size >= 2 and np.ptp(samples) > 0:
                try:
                    kde = gaussian_kde(samples, bw_method="scott")
                    raw = _kde_resample(kde, KDE_RESAMPLE_N, self.rng)
                    raw = np.clip(np.rint(raw), 1, MAX_EXAM_SCORE).astype(int)
                    hist = np.bincount(raw, minlength=MAX_EXAM_SCORE + 1)[1:].astype(np.float32)
                    self._exam_cdf[j] = np.cumsum(hist / hist.sum())
                    logger.debug("   exam %s: KDE ok (%d образцов, σ=%.2f)", eid, samples.size, samples.std(ddof=1))
                    continue
                except Exception:
                    pass
            self._exam_cdf[j] = self.global_cdf
            logger.debug("   exam %s: fallback → глобальная CDF", eid)

        # --- Места по программам --------------------------------------------
        self.seats_per_program = np.zeros(self.n_programs, np.int32)
        for p_code, seats in submission_stats[["program_code", "num_places"]].values:
            if p_code in self._program2idx:
                self.seats_per_program[self._program2idx[p_code]] = int(seats)

        # --- Диагностические/выходные коллекции -----------------------------
        self.admit_counter = np.zeros((self.n_applicants, self.n_programs), np.int32)
        self.pass_scores_collect = [[] for _ in range(self.n_programs)]
        self._apps_by_applicant: Dict[str, List[str]] = (
            applications.groupby("applicant_id")["program_code"].apply(list).to_dict()
        )
        self.global_sigma = float(global_samples.std(ddof=1)) if global_samples.size else 15.0

        # --- Заморозка нулей по истёкшим экзаменам (подготовка) -------------
        # Для каждой группы (applicant×exam) отметим freeze, если:
        #   • exam_freeze включен;
        #   • exam_id входит в expired_exam_ids;
        #   • в группе нет ни одной известной оценки (все vi==0).
        self._freeze_group: dict[tuple[int, int], bool] = {}
        frozen_groups = 0
        frozen_rows_total = 0
        if self.freeze_expired_exams and n_expired_ids > 0:
            for (a_idx, e_idx), rows in self._rows_by_app_exam.items():
                has_known = (self.vi_score[rows] > 0).any()
                to_freeze = bool(self.expired_exam_mask[e_idx] and not has_known)
                self._freeze_group[(a_idx, e_idx)] = to_freeze
                if to_freeze:
                    frozen_groups += 1
                    frozen_rows_total += int(rows.size)
        else:
            for k in self._rows_by_app_exam.keys():
                self._freeze_group[k] = False

        logger.info(
            "   истёкших exam_id=%d; замороженных групп a×exam=%d; затронуто строк=%d.",
            n_expired_ids, frozen_groups, frozen_rows_total
        )

    # --------------------------------------------------------------------- #
    def _single_simulation(self) -> None:
        vi = self.vi_score.copy()

        # Импутация по группам (applicant×exam): единый балл на группу.
        for (a_idx, e_idx), rows in self._rows_by_app_exam.items():
            if (vi[rows] == 0).any():
                existing = vi[rows][vi[rows] > 0]
                if existing.size:
                    # В группе есть реальная оценка → копируем её всем нулевым
                    vi[rows] = existing[0]
                    continue

                # Заморозка: если экзамен истёк и баллов в группе нет — оставляем нули
                if self._freeze_group.get((a_idx, e_idx), False):
                    continue

                # Иначе имитируем
                if self.personal_mu[a_idx] > 0:
                    score = self.rng.normal(self.personal_mu[a_idx], self.global_sigma)
                    score = int(np.clip(np.rint(score), 1, MAX_EXAM_SCORE))
                else:
                    u = float(self.rng.random())
                    score = 1 + int(np.searchsorted(self._exam_cdf[e_idx], u))
                vi[rows] = score

        total = (vi + self.id_ach).astype(np.int16)
        jitter = self.rng.random(self._rows).astype(np.float32)

        admitted, passing = _simulate_admission_numba(
            self.priority, self.program_idx, self.applicant_idx,
            total, self.seats_per_program, jitter,
            max_priority=int(self.priority.max()),
        )

        # Счётчики поступлений по программам
        for a_idx, p_idx in enumerate(admitted):
            if p_idx != -1:
                self.admit_counter[a_idx, p_idx] += 1

        # Проходные баллы
        for p_idx, scr in enumerate(passing):
            if scr != -1:
                self.pass_scores_collect[p_idx].append(int(scr))

    # --------------------------------------------------------------------- #
    def run_simulation(self) -> None:
        logger.info("Monte-Carlo: запускаем %d итераций…", self.n_sim)
        for i in range(self.n_sim):
            self._single_simulation()
            if (i + 1) % 500 == 0 or i == self.n_sim - 1:
                logger.debug("… %d / %d готово", i + 1, self.n_sim)

        self.prob_matrix = self.admit_counter / self.n_sim

        # Вероятности по программам
        self.p_admit = {
            aid: {
                code: float(self.prob_matrix[self._applicant2idx[aid], self._program2idx[code]])
                for code in self._apps_by_applicant.get(aid, [])
            }
            for aid in self._applicant2idx
        }

        # Квантили проходного (только по программам, где были наборы мест)
        self.pass_score_quantiles = {
            p_code: {
                "q90": float(np.percentile(scores, 90)),
                "q95": float(np.percentile(scores, 95)),
            }
            for p_code, p_idx in self._program2idx.items()
            if (scores := np.asarray(self.pass_scores_collect[p_idx])).size
        }

        # Диагностика в «докапаутной» трактовке:
        #   p_excluded = 0 (никого не исключали),
        #   p_fail_when_included = доля «не поступил» среди всех прогонов.
        admitted_totals = self.admit_counter.sum(axis=1)  # по applicant: число прогонов с зачислением
        self.diag = {
            aid: {
                "p_excluded": 0.0,
                "p_fail_when_included": float(1.0 - (admitted_totals[self._applicant2idx[aid]] / self.n_sim)),
            }
            for aid in self._applicant2idx
        }

        logger.info(
            "Monte-Carlo завершён: %d абитуриентов; %d направлений с квантилями. (без opt-out, freeze=%s)",
            len(self.p_admit),
            len(self.pass_score_quantiles),
            "ON" if self.freeze_expired_exams else "OFF",
        )

    # ------------------------------ API --------------------------------- #
    def get_probabilities(self) -> Dict[str, Dict[str, float]]:
        return self.p_admit

    def get_passing_score_quantiles(self) -> Dict[str, Dict[str, float]]:
        return self.pass_score_quantiles

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """{applicant_id: {'p_excluded': ..., 'p_fail_when_included': ...}}"""
        return self.diag
