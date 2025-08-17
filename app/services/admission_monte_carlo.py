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
    """
    Безопасная выборка из gaussian_kde для случаев, когда random_state недоступен.
    """
    try:
        return kde.resample(n, random_state=rng).ravel()
    except TypeError:
        # старые SciPy: нет random_state → временно «подменим» глобальный seed
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
    MC-модель шансов зачисления:
      • Импутация баллов ВИ (по personal μ, либо по CDF по экзамену, либо глобальная CDF).
      • Опциональный «opt-out» (исключение части абитуриентов без согласий).
      • «Заморозка нулей» по истёкшим экзаменам:
          если для пары (абитуриент × exam_id) все vi=0, и exam_id помечен как истёкший,
          то нули остаются нулями навсегда (импутация НЕ выполняется).
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

        # Настройки opt-out
        self.opt_out_enabled = settings.opt_out_enabled
        self.opt_out_ratio = float(settings.opt_out_ratio)
        self.opt_out_alpha = float(settings.opt_out_alpha)
        self.opt_out_mode = settings.opt_out_mode

        # Freeze экзаменов
        self.freeze_expired_exams = (
            settings.exam_freeze_enabled if freeze_expired_exams is None else bool(freeze_expired_exams)
        )
        self._expired_exam_ids_in = set(expired_exam_ids or [])

        logger.info(
            "AdmissionMonteCarlo: подготовка данных… "
            "(opt-out: %s, ratio=%.3f, alpha=%.2f, mode=%s; exam-freeze: %s)",
            "ON" if self.opt_out_enabled else "OFF",
            self.opt_out_ratio,
            self.opt_out_alpha,
            self.opt_out_mode,
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
        self.vi_score = applications[SCORE_COL].to_numpy(np.int16, copy=False)
        self.id_ach = applications["id_achievements"].to_numpy(np.int16, copy=False)
        self.consent = applications["consent"].to_numpy(bool, copy=False)

        # Ряды по (applicant, exam) и по applicant
        self._rows_by_app_exam: Dict[Tuple[int, int], np.ndarray] = {}
        self._rows_by_applicant: Dict[int, np.ndarray] = {}
        for r, (a, e) in enumerate(zip(self.applicant_idx, self.exam_idx)):
            self._rows_by_app_exam.setdefault((a, e), []).append(r)
            self._rows_by_applicant.setdefault(a, []).append(r)
        for k in list(self._rows_by_app_exam):
            self._rows_by_app_exam[k] = np.asarray(self._rows_by_app_exam[k], dtype=np.int32)
        for k in list(self._rows_by_applicant):
            self._rows_by_applicant[k] = np.asarray(self._rows_by_applicant[k], dtype=np.int32)

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
        # NB: предполагаем, что глобальных образцов достаточно.
        # (В реальных данных это верно; иначе можно ввести равномерную CDF.)
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
            # быстрое «все False», чтобы не плодить ветвлений в симуляции
            for k in self._rows_by_app_exam.keys():
                self._freeze_group[k] = False

        logger.info(
            "   истёкших exam_id=%d; замороженных групп a×exam=%d; затронуто строк=%d.",
            n_expired_ids, frozen_groups, frozen_rows_total
        )

        # --- Пул opt-out: кандидаты E и веса --------------------------------
        any_unconsented = np.zeros(self.n_applicants, dtype=bool)
        for a_idx in range(self.n_applicants):
            rows = self._rows_by_applicant.get(a_idx, [])
            if rows.size:
                any_unconsented[a_idx] = (~self.consent[rows]).any()

        # «Сила» абитуриента: максимум текущего vi_score по его заявкам (0 если неизвестно)
        max_vi = np.zeros(self.n_applicants, dtype=np.int16)
        for a_idx, rows in self._rows_by_applicant.items():
            max_vi[a_idx] = int(self.vi_score[rows].max(initial=0))

        # Ранги в [0,1] (избегаем нулевых весов)
        if max_vi.max() > max_vi.min():
            ranks = (max_vi - max_vi.min()) / (max_vi.max() - max_vi.min())
        else:
            ranks = np.zeros_like(max_vi, dtype=np.float32)
        base_w = np.power(ranks, self.opt_out_alpha).astype(np.float64)
        base_w += 1e-9  # защита от нулей
        self._opt_pool_mask = any_unconsented
        self._opt_weights = base_w

        # Диагностика, коллекции результатов
        self.excluded_counter = np.zeros(self.n_applicants, np.int32)      # сколько раз был исключён
        self.fail_included_counter = np.zeros(self.n_applicants, np.int32) # среди включённых — не поступил
        self._last_sampled_excluded = None

        self.admit_counter = np.zeros((self.n_applicants, self.n_programs), np.int32)
        self.pass_scores_collect = [[] for _ in range(self.n_programs)]
        self._apps_by_applicant: Dict[str, List[str]] = (
            applications.groupby("applicant_id")["program_code"].apply(list).to_dict()
        )
        self.global_sigma = float(global_samples.std(ddof=1)) if global_samples.size else 15.0

        # Если opt-out=ON и есть пул — логируем размер
        if self.opt_out_enabled:
            E = int(self._opt_pool_mask.sum())
            K = int(np.floor(self.opt_out_ratio * E)) if E > 0 else 0
            logger.info("→ Opt-out пул (E): %d абитуриентов; будем исключать K=%d (%.1f%%).",
                        E, K, self.opt_out_ratio * 100.0)

        # Предвыборка «once»
        self._excluded_once: np.ndarray | None = None
        if self.opt_out_enabled and self.opt_out_mode == "once" and self._opt_pool_mask.any():
            self._excluded_once = self._sample_excluded()

    # --------------------------------------------------------------------- #
    def _sample_excluded(self) -> np.ndarray:
        pool_idx = np.where(self._opt_pool_mask)[0]
        if pool_idx.size == 0 or not self.opt_out_enabled:
            return np.array([], dtype=np.int32)
        K = int(np.floor(self.opt_out_ratio * pool_idx.size))
        if K <= 0:
            return np.array([], dtype=np.int32)
        w = self._opt_weights[pool_idx]
        w = w / w.sum()
        excluded = self.rng.choice(pool_idx, size=K, replace=False, p=w).astype(np.int32)
        return excluded

    def _single_simulation(self) -> None:
        vi = self.vi_score.copy()
        pr = self.priority.copy()

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

        # Opt-out: исключаем часть абитуриентов из пула E
        excluded = np.array([], dtype=np.int32)
        if self.opt_out_enabled and self._opt_pool_mask.any():
            if self._excluded_once is not None:
                excluded = self._excluded_once
            else:
                excluded = self._sample_excluded()

            if excluded.size > 0:
                max_prio = int(pr.max())
                for a_idx in excluded:
                    rows = self._rows_by_applicant.get(int(a_idx), None)
                    if rows is not None and rows.size:
                        pr[rows] = max_prio + 1  # выводим за пределы цикла приоритетов
                self.excluded_counter[excluded] += 1

                # редкий DEBUG-лог
                if self.rng.random() < 0.003:
                    ex = int(excluded[0])
                    self._last_sampled_excluded = ex
                    logger.debug("Отток в симуляции: исключены %d абитуриентов (пример a_idx=%s…)", excluded.size, ex)

        total = (vi + self.id_ach).astype(np.int16)
        jitter = self.rng.random(self._rows).astype(np.float32)

        admitted, passing = _simulate_admission_numba(
            pr, self.program_idx, self.applicant_idx,
            total, self.seats_per_program, jitter,
            max_priority=int(pr.max()),
        )

        # Счётчики поступлений по программам
        for a_idx, p_idx in enumerate(admitted):
            if p_idx != -1:
                self.admit_counter[a_idx, p_idx] += 1

        # Диагностика «провалился среди включённых»
        if self.opt_out_enabled and self._opt_pool_mask.any():
            incl_mask = np.ones(self.n_applicants, dtype=bool)
            incl_mask[excluded] = False
            fail_mask = (admitted == -1) & incl_mask
            self.fail_included_counter[fail_mask] += 1
        else:
            # Если opt-out OFF, то «включён» = все
            fail_mask = (admitted == -1)
            self.fail_included_counter[fail_mask] += 1

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

        # Диагностика: отток и «пролёты»
        p_excluded = self.excluded_counter / self.n_sim
        included_cnt = np.maximum(self.n_sim - self.excluded_counter, 1)  # защита от деления на ноль
        p_fail_included = self.fail_included_counter / included_cnt

        self.diag = {
            aid: {
                "p_excluded": float(p_excluded[self._applicant2idx[aid]]),
                "p_fail_when_included": float(p_fail_included[self._applicant2idx[aid]]),
            }
            for aid in self._applicant2idx
        }

        logger.info(
            "Monte-Carlo завершён: %d абитуриентов; %d направлений с квантилями. "
            "Opt-out: %s, пул=%d, режим=%s",
            len(self.p_admit),
            len(self.pass_score_quantiles),
            "ON" if self.opt_out_enabled else "OFF",
            int(self._opt_pool_mask.sum()),
            self.opt_out_mode,
        )

    # ------------------------------ API --------------------------------- #
    def get_probabilities(self) -> Dict[str, Dict[str, float]]:
        return self.p_admit

    def get_passing_score_quantiles(self) -> Dict[str, Dict[str, float]]:
        return self.pass_score_quantiles

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """{applicant_id: {'p_excluded': ..., 'p_fail_when_included': ...}}"""
        return self.diag
