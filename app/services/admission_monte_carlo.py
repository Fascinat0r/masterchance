# app/services/admission_monte_carlo.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Monte‑Carlo‑модель конкурса в магистратуру.

В этой ревизии:
• экзамен генерируется единожды для пары (абитуриент, exam_id) и
  вставляется во все заявки этой пары.
"""
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import gaussian_kde

from app.config.logger import logger


# ───────────────────────  КОНСТАНТЫ  ───────────────────────
MAX_EXAM_SCORE      = 100
MAX_ID_ACHIEVEMENTS = 10
MAX_TOTAL_SCORE     = MAX_EXAM_SCORE + MAX_ID_ACHIEVEMENTS
SCORE_COL           = "vi_score"

KDE_RESAMPLE_N = 10_000
RANK_SCALE     = 100               # масштаб джиттера в ядре


# ──────────────────────────  HELPERS  ──────────────────────
def _kde_resample(kde: gaussian_kde, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Совместимый ресэмплинг для старых и новых версий SciPy."""
    try:                               # SciPy 1.11+
        return kde.resample(n, random_state=rng).ravel()
    except TypeError:                  # SciPy ≤ 1.10
        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        saved_state = np.random.get_state()
        try:
            np.random.seed(seed)
            return kde.resample(n).ravel()
        finally:
            np.random.set_state(saved_state)


# ──────────────────────  NUMBA‑ядро конкурса  ──────────────
@njit(cache=True)
def _simulate_admission_numba(priority, program_idx, applicant_idx,
                              total_score, seats_init, jitter,
                              max_priority) -> Tuple[np.ndarray, np.ndarray]:
    A = applicant_idx.max() + 1
    P = seats_init.size
    N = priority.size

    admitted = np.full(A, -1, np.int32)
    passing  = np.full(P, -1, np.int16)

    max_seats   = np.max(seats_init)
    seat_cnt    = np.zeros(P, np.int32)
    seats_left  = seats_init.copy()

    tab_app   = np.full((P, max_seats), -1, np.int32)
    tab_score = np.full((P, max_seats), -1, np.int16)
    tab_rank  = np.full((P, max_seats),  0, np.int32)

    worst_score = np.full(P, -1, np.int16)
    worst_rank  = np.full(P, -1, np.int32)
    worst_slot  = np.zeros(P, np.int32)

    for pr in range(1, max_priority + 1):
        for i in range(N):
            if priority[i] != pr:
                continue
            appl = applicant_idx[i]
            if admitted[appl] != -1:
                continue

            prog = program_idx[i]
            raw  = np.int16(total_score[i])
            rank = raw * RANK_SCALE + np.int32(jitter[i] * RANK_SCALE)

            # свободное место
            if seats_left[prog] > 0:
                s = seat_cnt[prog]
                tab_app[prog, s]   = appl
                tab_score[prog, s] = raw
                tab_rank[prog, s]  = rank

                seat_cnt[prog]   += 1
                seats_left[prog] -= 1
                admitted[appl]    = prog

                if worst_rank[prog] == -1 or rank < worst_rank[prog]:
                    worst_rank[prog]  = rank
                    worst_score[prog] = raw
                    worst_slot[prog]  = s
                continue

            # вытеснение
            if rank > worst_rank[prog]:
                kick     = worst_slot[prog]
                old_appl = tab_app[prog, kick]
                admitted[old_appl] = -1

                tab_app[prog, kick]   = appl
                tab_score[prog, kick] = raw
                tab_rank[prog, kick]  = rank
                admitted[appl]        = prog

                wr, ws, wslt = rank, raw, kick
                for t in range(seat_cnt[prog]):
                    r = tab_rank[prog, t]
                    if r < wr:
                        wr, ws, wslt = r, tab_score[prog, t], t
                worst_rank[prog]  = wr
                worst_score[prog] = ws
                worst_slot[prog]  = wslt

    # проходные баллы
    for p in range(P):
        if seat_cnt[p] > 0:
            passing[p] = worst_score[p]

    return admitted, passing


# ───────────────────────  ГЛАВНЫЙ КЛАСС  ──────────────────
class AdmissionMonteCarlo:
    """MC‑оценка шансов поступления (1 экзамен + ИД)."""

    # ────────────  INIT  ────────────
    def __init__(self,
                 applications:    pd.DataFrame,
                 applicants:      pd.DataFrame | None,
                 submission_stats: pd.DataFrame,
                 programs_meta:   pd.DataFrame,
                 *,
                 n_simulations: int = 10_000,
                 random_seed:   int | None = None):
        self.n_sim = n_simulations
        self.rng   = np.random.default_rng(random_seed)

        logger.info("AdmissionMonteCarlo: подготовка данных…")

        self._rows = len(applications)

        # --- индексы ---------------------------------------------------
        self._applicant2idx = {aid: i
                               for i, aid in
                               enumerate(applications["applicant_id"].unique())}
        self._program2idx   = {c: i
                               for i, c in
                               enumerate(applications["program_code"].unique())}
        self.n_applicants = len(self._applicant2idx)
        self.n_programs   = len(self._program2idx)

        # --- exam_id ---------------------------------------------------
        logger.info("→ определяем exam_id (кафедра / __eng)…")
        meta = programs_meta.set_index("program_code")
        self.exam_id = np.empty(self._rows, dtype="U16")
        for i, p_code in enumerate(applications["program_code"]):
            row  = meta.loc[p_code]
            dept = row["department_code"]
            self.exam_id[i] = f"{dept}__eng" if row["is_international"] else dept

        self._exam2idx = {eid: j for j, eid in enumerate(np.unique(self.exam_id))}
        self.exam_idx  = np.vectorize(self._exam2idx.get)(self.exam_id).astype(np.int32)
        self.n_exams   = len(self._exam2idx)
        logger.info("   найдено %d различных экзаменов.", self.n_exams)

        # --- NumPy‑векторизация ---------------------------------------
        self.applicant_idx = applications["applicant_id"].map(
            self._applicant2idx).to_numpy(np.int32, copy=False)
        self.program_idx = applications["program_code"].map(
            self._program2idx).to_numpy(np.int32, copy=False)
        self.priority = applications["priority"].to_numpy(np.int16, copy=False)
        self.vi_score = applications[SCORE_COL].to_numpy(np.int16, copy=False)
        self.id_ach   = applications["id_achievements"].to_numpy(np.int16, copy=False)

        # --- индекс «строки по (applicant, exam)» ----------------------
        self._rows_by_app_exam: Dict[Tuple[int, int], np.ndarray] = {}
        for r, (a, e) in enumerate(zip(self.applicant_idx, self.exam_idx)):
            self._rows_by_app_exam.setdefault((a, e), []).append(r)
        # превращаем списки в массивы int32
        for k, v in list(self._rows_by_app_exam.items()):
            self._rows_by_app_exam[k] = np.asarray(v, dtype=np.int32)

        # --- personal μ -----------------------------------------------
        logger.info("→ вычисляем personal_mu …")
        known = (self.vi_score > 0) & (self.vi_score < MAX_EXAM_SCORE)
        sums  = np.bincount(self.applicant_idx[known],
                            self.vi_score[known].astype(np.float32),
                            minlength=self.n_applicants)
        cnts  = np.bincount(self.applicant_idx[known], minlength=self.n_applicants)
        self.personal_mu = np.zeros(self.n_applicants, np.float32)
        mask = cnts > 0
        self.personal_mu[mask] = sums[mask] / cnts[mask]
        logger.debug("   персональный μ есть у %d / %d абитуриентов.",
                     mask.sum(), self.n_applicants)

        # --- KDE → CDF --------------------------------------------------
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
                    kde  = gaussian_kde(samples, bw_method="scott")
                    raw  = _kde_resample(kde, KDE_RESAMPLE_N, self.rng)
                    raw  = np.clip(np.rint(raw), 1, MAX_EXAM_SCORE).astype(int)
                    hist = np.bincount(raw, minlength=MAX_EXAM_SCORE + 1)[1:].astype(np.float32)
                    self._exam_cdf[j] = np.cumsum(hist / hist.sum())
                    logger.debug("   exam %s: KDE ok (%d образцов, σ=%.2f)",
                                 eid, samples.size, samples.std(ddof=1))
                    continue
                except Exception:
                    pass   # fallback ниже

            self._exam_cdf[j] = self.global_cdf
            logger.debug("   exam %s: fallback → глобальная CDF", eid)

        # --- места ------------------------------------------------------
        self.seats_per_program = np.zeros(self.n_programs, np.int32)
        for p_code, seats in submission_stats[["program_code", "num_places"]].values:
            if p_code in self._program2idx:
                self.seats_per_program[self._program2idx[p_code]] = int(seats)

        # --- выводные счётчики -----------------------------------------
        self.admit_counter      = np.zeros((self.n_applicants, self.n_programs), np.int32)
        self.pass_scores_collect = [[] for _ in range(self.n_programs)]

        self._apps_by_applicant: Dict[str, List[str]] = (
            applications.groupby("applicant_id")["program_code"].apply(list).to_dict()
        )

        self.global_sigma = float(global_samples.std(ddof=1))

    # ──────────  ОДИН ПРОГОН  ──────────
    def _single_simulation(self) -> None:
        vi = self.vi_score.copy()

        # --- заполняем пропуски единым баллом на (applicant, exam) ------
        for (a_idx, e_idx), rows in self._rows_by_app_exam.items():
            if (vi[rows] == 0).any():
                # уже есть сданный балл в группе?
                existing = vi[rows][vi[rows] > 0]
                if existing.size:
                    vi[rows] = existing[0]
                    continue

                # личная история
                if self.personal_mu[a_idx] > 0:
                    score = self.rng.normal(self.personal_mu[a_idx], self.global_sigma)
                    score = int(np.clip(np.rint(score), 1, MAX_EXAM_SCORE))
                else:
                    u = float(self.rng.random())
                    score = 1 + int(np.searchsorted(self._exam_cdf[e_idx], u))

                vi[rows] = score

        # --- остальные этапы -------------------------------------------
        total  = (vi + self.id_ach).astype(np.int16)
        jitter = self.rng.random(self._rows).astype(np.float32)

        admitted, passing = _simulate_admission_numba(
            self.priority, self.program_idx, self.applicant_idx,
            total, self.seats_per_program, jitter,
            max_priority=int(self.priority.max()),
        )

        for a_idx, p_idx in enumerate(admitted):
            if p_idx != -1:
                self.admit_counter[a_idx, p_idx] += 1

        for p_idx, scr in enumerate(passing):
            if scr != -1:
                self.pass_scores_collect[p_idx].append(int(scr))

    # ──────────  RUN  ──────────
    def run_simulation(self) -> None:
        logger.info("Monte‑Carlo: запускаем %d итераций…", self.n_sim)
        for i in range(self.n_sim):
            self._single_simulation()
            if (i + 1) % 500 == 0 or i == self.n_sim - 1:
                logger.debug("… %d / %d готово", i + 1, self.n_sim)

        self.prob_matrix = self.admit_counter / self.n_sim

        self.p_admit = {
            aid: {
                code: float(self.prob_matrix[self._applicant2idx[aid],
                                             self._program2idx[code]])
                for code in self._apps_by_applicant.get(aid, [])
            }
            for aid in self._applicant2idx
        }

        self.pass_score_quantiles = {
            p_code: {
                "q90": float(np.percentile(scores, 90)),
                "q95": float(np.percentile(scores, 95)),
            }
            for p_code, p_idx in self._program2idx.items()
            if (scores := np.asarray(self.pass_scores_collect[p_idx])).size
        }

        logger.info("Monte‑Carло завершён: %d абитуриентов; "
                    "%d направлений с квантилями.",
                    len(self.p_admit), len(self.pass_score_quantiles))

    # ──────────  API  ──────────
    def get_probabilities(self) -> Dict[str, Dict[str, float]]:
        return self.p_admit

    def get_passing_score_quantiles(self) -> Dict[str, Dict[str, float]]:
        return self.pass_score_quantiles
