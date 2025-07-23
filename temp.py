from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt

from app.config.config import settings
from app.infrastructure.db.queries.statistics import (
    top_n_competition,
    bottom_n_competition,
    programs_with_free_places,
    applicant_with_most_applications,
    top_programs_by_avg_score, total_places, total_places_non_ino, total_applications, count_exam_submitted,
    subject1_score_distribution
)


def plot_subject1_score_distribution(scores, show=True, save_path=None):
    """
    Рисует гистограмму переданных баллов.
    """
    if not scores:
        print("Нет данных для построения гистограммы.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=range(0, 101, 5), edgecolor="black", align="left")
    plt.title("Распределение баллов за вступительное испытание (предмет 1)")
    plt.xlabel("Балл")
    plt.ylabel("Количество заявок")
    plt.grid(axis="y")
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


engine = create_engine(settings.database_url, echo=settings.db_echo, future=True)
Session = sessionmaker(bind=engine, future=True)
session = Session()

print("Топ‑5 конкуренции:", top_n_competition(session))
print("5 наименьших конкуренций:", bottom_n_competition(session))
print("Программы с свободными местами:", programs_with_free_places(session))
print("Самый активный абитуриент:", applicant_with_most_applications(session))
print("Топ‑10 по среднему баллу:", top_programs_by_avg_score(session))

print("Всего мест:", total_places(session))
print("Всего мест (без ИНО):", total_places_non_ino(session))

total_apps = total_applications(session)
exam_submitted = count_exam_submitted(session)
pct = (exam_submitted / total_apps * 100) if total_apps else 0.0

print(f"Заявок с ненулевыми баллами за экзамены: {exam_submitted} из {total_apps} ({pct:.1f}%)")

scores, distr, mean, median, var = subject1_score_distribution(session)
print("\nРаспределение баллов (только 1-99):")
for rng, pct in distr.items():
    print(f"{rng}: {pct}%")
print(f"Среднее: {mean:.2f}, медиана: {median:.2f}, дисперсия: {var:.2f}")
plot_subject1_score_distribution(scores)

session.close()
