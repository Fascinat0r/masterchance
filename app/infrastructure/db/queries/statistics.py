# app/application/statistics.py

from typing import List, Tuple

import numpy as np
from sqlalchemy import func, desc, asc
from sqlalchemy.orm import Session

from app.infrastructure.db.models import (
    ProgramModel,
    SubmissionStatsModel,
    ApplicationModel
)


def top_n_competition(
        session: Session,
        limit: int = 5
) -> List[Tuple[str, float]]:
    """
    Топ N программ по конкуренции (заявок на место), игнорируя is_ino.
    Возвращает список кортежей (program_code, competition_ratio).
    """
    q = (
        session.query(
            SubmissionStatsModel.program_code,
            (SubmissionStatsModel.num_applications / SubmissionStatsModel.num_places)
            .label("competition")
        )
        .join(ProgramModel, ProgramModel.code == SubmissionStatsModel.program_code)
        .filter(ProgramModel.is_ino.is_(False))
        .order_by(desc("competition"))
        .limit(limit)
    )
    return q.all()


def bottom_n_competition(
        session: Session,
        limit: int = 5
) -> List[Tuple[str, float]]:
    """
    Топ N программ с наименьшей конкуренцией (заявок на место), игнорируя is_ino.
    """
    q = (
        session.query(
            SubmissionStatsModel.program_code,
            (SubmissionStatsModel.num_applications / SubmissionStatsModel.num_places)
            .label("competition")
        )
        .join(ProgramModel, ProgramModel.code == SubmissionStatsModel.program_code)
        .filter(
            ProgramModel.is_ino.is_(False),
            SubmissionStatsModel.num_applications < SubmissionStatsModel.num_places
        )
        .order_by(asc("competition"))
        .limit(limit)
    )
    return q.all()


def programs_with_free_places(
        session: Session
) -> List[str]:
    """
    Список всех программ (codes) без is_ino, где свободные места:
    num_applications < num_places.
    """
    q = (
        session.query(SubmissionStatsModel.program_code)
        .join(ProgramModel, ProgramModel.code == SubmissionStatsModel.program_code)
        .filter(
            ProgramModel.is_ino.is_(False),
            SubmissionStatsModel.num_applications < SubmissionStatsModel.num_places
        )
    )
    return [row.program_code for row in q.all()]


def applicant_with_most_applications(
        session: Session
) -> Tuple[str, int]:
    """
    Абитуриент, подавший наибольшее число заявок.
    Возвращает (applicant_id, applications_count).
    """
    q = (
        session.query(
            ApplicationModel.applicant_id,
            func.count().label("cnt")
        )
        .group_by(ApplicationModel.applicant_id)
        .order_by(desc("cnt"))
        .limit(1)
    )
    return q.one()


def top_programs_by_avg_score(
        session: Session,
        limit: int = 10
) -> List[Tuple[str, float]]:
    """
    Топ N программ по среднему total_score, игнорируя is_ino.
    Возвращает [(program_code, avg_score), ...].
    """
    q = (
        session.query(
            ApplicationModel.program_code,
            func.avg(ApplicationModel.total_score).label("avg_score")
        )
        .join(ProgramModel, ProgramModel.code == ApplicationModel.program_code)
        .filter(ProgramModel.is_ino.is_(False))
        .group_by(ApplicationModel.program_code)
        .order_by(desc("avg_score"))
        .limit(limit)
    )
    return q.all()


def total_places(
        session: Session
) -> int:
    """
    Общее число мест по всем направлениям.
    """
    q = session.query(func.sum(SubmissionStatsModel.num_places).label("sum_places"))
    return q.scalar() or 0


def total_places_non_ino(
        session: Session
) -> int:
    """
    Общее число мест по всем направлениям, игнорируя is_ino.
    """
    q = (
        session.query(func.sum(SubmissionStatsModel.num_places).label("sum_places"))
        .join(ProgramModel, ProgramModel.code == SubmissionStatsModel.program_code)
        .filter(ProgramModel.is_ino.is_(False))
    )
    return q.scalar() or 0


def count_exam_submitted(
        session: Session
) -> int:
    """
    Число заявок, у которых оба экзаменационных балла > 0.
    """
    q = (
        session.query(func.count())
        .select_from(ApplicationModel)
        .filter(
            ApplicationModel.subject1_score > 0
        )
    )
    return q.scalar() or 0


def total_applications(
        session: Session
) -> int:
    """
    Общее число заявок.
    """
    q = session.query(func.count()).select_from(ApplicationModel)
    return q.scalar() or 0


def subject1_score_distribution(session):
    """
    Возвращает список subject1_score (без 0 и 100), а также считает распределение по диапазонам, среднее, медиану, дисперсию.
    """
    # Получить все подходящие баллы
    scores = [
        s for (s,) in session.query(ApplicationModel.subject1_score)
        .join(ProgramModel, ProgramModel.code == ApplicationModel.program_code)
        .filter(
            ProgramModel.is_ino.is_(False),
            ApplicationModel.subject1_score > 0,
            ApplicationModel.subject1_score < 100
        )
        .all()
    ]
    if not scores:
        return [], {}, None, None, None

    # Разбивка по диапазонам
    bins = list(range(0, 101, 10))
    hist, _ = np.histogram(scores, bins=bins)
    ranges = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]
    total = sum(hist)
    distr = {ranges[i]: round(hist[i] / total * 100, 1) for i in range(len(hist))}

    # Статистики
    mean = np.mean(scores)
    median = np.median(scores)
    var = np.var(scores)

    return scores, distr, mean, median, var
