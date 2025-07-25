# app/services/admission_monte_carlo.py
from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from numba import njit

from app.config.logger import logger

# ───────────────────────────────  CONSTANTS  ────────────────────────────────
MAX_EXAM_SCORE = 100  # «ВИ» оценивается 0‑100
MAX_ID_ACHIEVEMENTS = 10  # индивидуальные достижения 0‑10
MAX_TOTAL_SCORE = MAX_EXAM_SCORE + MAX_ID_ACHIEVEMENTS  # 110
SCORE_COL = "vi_score"  # единственный экзамен


# ──────────────────────────────  NUMBA KERNEL  ──────────────────────────────
@njit(cache=True)
def _simulate_admission_numba(
        priority, program_idx, applicant_idx,
        total_score,
        seats_init,
        jitter,                     # (N,) 0‑1
        max_priority
) -> Tuple[np.ndarray, np.ndarray]:
    A = applicant_idx.max() + 1
    P = seats_init.size
    N = priority.size

    admitted = np.full(A, -1, np.int32)
    passing  = np.full(P, -1, np.int16)

    max_seats  = np.max(seats_init)
    seat_cnt   = np.zeros(P, np.int32)
    seats_left = seats_init.copy()

    # двойные таблицы
    tab_app   = np.full((P, max_seats), -1, np.int32)
    tab_score = np.full((P, max_seats), -1, np.int16)   # raw балл
    tab_rank  = np.full((P, max_seats),  0, np.int32)   # ранговый

    worst_score = np.full(P, -1,  np.int16)   # худший raw
    worst_rank  = np.full(P, -1,  np.int32)   # его rank
    worst_slot  = np.zeros(P, np.int32)

    for pr in range(1, max_priority + 1):

        for i in range(N):
            if priority[i] != pr:
                continue
            appl = applicant_idx[i]
            if admitted[appl] != -1:
                continue

            prog    = program_idx[i]
            raw     = np.int16(total_score[i])
            rank    = raw * 100 + np.int32(jitter[i] * 100)   # 0‑99 добавка

            # -- ещё свободно -------------------------------------------------
            if seats_left[prog] > 0:
                s = seat_cnt[prog]
                tab_app[prog, s]   = appl
                tab_score[prog, s] = raw
                tab_rank[prog, s]  = rank

                seat_cnt[prog]    += 1
                seats_left[prog]  -= 1
                admitted[appl]    = prog

                if worst_rank[prog] == -1 or rank < worst_rank[prog]:
                    worst_rank[prog]  = rank
                    worst_score[prog] = raw
                    worst_slot[prog]  = s
                continue

            # -- нет мест: проверяем вытеснение -------------------------------
            if rank > worst_rank[prog]:
                kick = worst_slot[prog]
                old_appl = tab_app[prog, kick]
                admitted[old_appl] = -1          # вылетел

                # ставим нового
                tab_app[prog, kick]   = appl
                tab_score[prog, kick] = raw
                tab_rank[prog,  kick] = rank
                admitted[appl]        = prog

                # ищем нового худшего
                wr  = rank
                ws  = raw
                wsl = kick
                for t in range(seat_cnt[prog]):
                    r = tab_rank[prog, t]
                    if r < wr:
                        wr  = r
                        ws  = tab_score[prog, t]
                        wsl = t
                worst_rank[prog]  = wr
                worst_score[prog] = ws
                worst_slot[prog]  = wsl

    # --- проходные ----------------------------------------------------------
    for p in range(P):
        if seat_cnt[p] > 0:
            passing[p] = worst_score[p]

    return admitted, passing

# ─────────────────────────────  MAIN CLASS  ────────────────────────────────
class AdmissionMonteCarlo:
    """
    Высокопроизводительная Монте‑Карло‑модель поступления
    (один экзамен `vi_score` + ИД).
    """

    # ────────── INIT ────────────────────────────────────────────────────────
    def __init__(self,
                 applications: pd.DataFrame,
                 applicants: pd.DataFrame | None,
                 submission_stats: pd.DataFrame,
                 *,
                 n_simulations: int = 10_000,
                 random_seed: int | None = None):
        self.n_sim = n_simulations
        self.rng = np.random.default_rng(random_seed)

        logger.info("AdmissionMonteCarlo: подготовка данных…")

        # ——— индексация -----------------------------------------------------
        self._applicant2idx = {aid: i
                               for i, aid in enumerate(applications["applicant_id"].unique())}
        self._program2idx = {p: i
                             for i, p in enumerate(applications["program_code"].unique())}

        self._apps_by_applicant: dict[str, list[str]] = (
            applications.groupby("applicant_id")["program_code"]
            .apply(list)
            .to_dict()
        )

        self.n_applicants = len(self._applicant2idx)
        self.n_programs = len(self._program2idx)
        self._rows = len(applications)

        # ——— NumPy‑векторизация заявок -------------------------------------
        self.applicant_idx = applications["applicant_id"].map(
            self._applicant2idx).to_numpy(np.int32, copy=False)
        self.program_idx = applications["program_code"].map(
            self._program2idx).to_numpy(np.int32, copy=False)

        self.priority = applications["priority"].to_numpy(np.int16, copy=False)
        self.vi_score = applications[SCORE_COL].to_numpy(np.int16, copy=False)
        self.id_ach = applications["id_achievements"].to_numpy(np.int16, copy=False)

        # пропуски экзамена
        self.missing_exam_idx = np.where(self.vi_score == 0)[0]

        # ——— статистика базовых средних ------------------------------------
        self._precompute_statistics(applications)

        # ——— квоты мест -----------------------------------------------------
        self.seats_per_program = np.zeros(self.n_programs, dtype=np.int32)
        for p_code, seats in submission_stats[["program_code", "num_places"]].values:
            if p_code in self._program2idx:
                self.seats_per_program[self._program2idx[p_code]] = int(seats)

        # ——— буферы результатов -------------------------------------------
        self.admit_counter = np.zeros((self.n_applicants, self.n_programs), dtype=np.int32)
        self.pass_scores_collect: List[List[int]] = [[] for _ in range(self.n_programs)]

    # ────────── PRE‑COMPUTE ────────────────────────────────────────────────
    def _precompute_statistics(self, df: pd.DataFrame) -> None:
        """Глобальное, персональное и программное средние по «ВИ»"""
        logger.info("→ вычисляем μ/σ (без 0 и 100)…")

        exam_vals = df.loc[(df[SCORE_COL] > 0) &
                           (df[SCORE_COL] < MAX_EXAM_SCORE), SCORE_COL].to_numpy(np.float32, copy=False)

        if exam_vals.size == 0:
            raise ValueError("Нет ни одного сданного ВИ (>0 и <100).")

        self.global_mu = float(exam_vals.mean())
        self.global_sigma = float(exam_vals.std(ddof=1))

        # персональные μ
        df["__tmp_mean"] = df[SCORE_COL].where(
            (df[SCORE_COL] > 0) & (df[SCORE_COL] < MAX_EXAM_SCORE))
        per_app_mu = df.groupby("applicant_id")["__tmp_mean"].mean().to_dict()

        # программные μ
        per_prog_mu = df.groupby("program_code")["__tmp_mean"].mean().to_dict()

        # μ для каждой строки заявки
        self.mu_row = np.vectorize(per_app_mu.get)(df["applicant_id"])
        mu_prog = np.vectorize(per_prog_mu.get)(df["program_code"])
        mask_nan = np.isnan(self.mu_row)
        self.mu_row[mask_nan] = mu_prog[mask_nan]
        self.mu_row[np.isnan(self.mu_row)] = self.global_mu
        self.mu_row = self.mu_row.astype(np.float32)

        # готово
        logger.debug("μ_global=%.2f, σ=%.2f | персональных=%d | программных=%d | rows=%d",
                     self.global_mu, self.global_sigma,
                     len(per_app_mu), len(per_prog_mu), self._rows)

        # сохраняем наружу (для вывода «базовых» метрик)
        self.per_applicant_mean = per_app_mu
        self.per_program_mean = per_prog_mu

    # ────────── ОДИН ПРОГОН ────────────────────────────────────────────────
    def _single_simulation(self) -> None:
        """
        • дорисовываем недостающие vi_score
        • считаем total_score (= vi + ИД)
        • применяем JIT‑ядро конкурса
        • инкрементируем счётчики результатов
        """
        vi = self.vi_score.copy()

        # 1. дописываем пропуски ------------------------------------------------
        if self.missing_exam_idx.size:
            gen = self.rng.normal(self.mu_row[self.missing_exam_idx], self.global_sigma)
            gen = np.clip(np.rint(gen), 1, MAX_EXAM_SCORE).astype(np.int16)
            vi[self.missing_exam_idx] = gen

        # 2. total = exam + ИД --------------------------------------------------
        total = (vi + self.id_ach).astype(np.int16)

        # контроль диапазона
        if (bad := total > MAX_TOTAL_SCORE).any():
            logger.warning("⛔ total_score выходит за пределы (>%d) у %d заявок. "
                           "Примеры: %s", MAX_TOTAL_SCORE, bad.sum(),
                           total[bad][:10])
            total[bad] = MAX_TOTAL_SCORE  # жёсткий клэмп; не должен включаться!

        # 3a) готовим массив случайного шума [0,1) для tiebreaking
        # используем тот же RNG, что и для пропусков экзамена
        jitter = self.rng.random(self._rows).astype(np.float32)

        # 3. конкурс -----------------------------------------------------------
        admitted, passing = _simulate_admission_numba(
            self.priority,
            self.program_idx,
            self.applicant_idx,
            total,
            self.seats_per_program,
            jitter,
            max_priority=int(self.priority.max()),
        )

        # 4. инкремент счётчиков ----------------------------------------------
        for a_idx, p_idx in enumerate(admitted):
            if p_idx != -1:
                self.admit_counter[a_idx, p_idx] += 1

        for p_idx, scr in enumerate(passing):
            if scr != -1:
                self.pass_scores_collect[p_idx].append(int(scr))

    # ────────── ГЛАВНЫЙ ЦИКЛ ────────────────────────────────────────────────
    def run_simulation(self) -> None:
        logger.info("Monte‑Carlo: запускаем %d итераций…", self.n_sim)
        for i in range(self.n_sim):
            self._single_simulation()
            if (i + 1) % 500 == 0 or i == self.n_sim - 1:
                logger.debug("… %d / %d готово", i + 1, self.n_sim)

        # 5. финальная агрегация ----------------------------------------------
        self.prob_matrix = self.admit_counter / self.n_sim

        # вероятности по студентам
        self.p_admit: dict[str, dict[str, float]] = {}
        for aid, a_idx in self._applicant2idx.items():
            row = self.prob_matrix[a_idx]
            codes = self._apps_by_applicant.get(aid, [])
            self.p_admit[aid] = {
                code: float(row[self._program2idx[code]]) if code in self._program2idx else 0.0
                for code in codes
            }

        # квантили проходных баллов
        self.pass_score_quantiles: Dict[str, Dict[str, float]] = {}
        for p_code, p_idx in self._program2idx.items():
            arr = np.array(self.pass_scores_collect[p_idx], dtype=np.int16)
            if arr.size:
                self.pass_score_quantiles[p_code] = {
                    "q90": float(np.percentile(arr, 90)),
                    "q95": float(np.percentile(arr, 95)),
                }

        logger.info("Monte‑Carло окончен. "
                    "Вероятности рассчитаны для %d абитуриентов, "
                    "%d направлений имеют непустые квантили.",
                    len(self.p_admit), len(self.pass_score_quantiles))

    # ────────── API ──────────────────────────────────────────────────────────
    def get_probabilities(self) -> Dict[str, Dict[str, float]]:
        """applicant_id → {program_code: p}"""
        return self.p_admit

    def get_passing_score_quantiles(self) -> Dict[str, Dict[str, float]]:
        """program_code → {'q90': …, 'q95': …}"""
        return self.pass_score_quantiles
