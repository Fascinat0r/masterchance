from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config.config import settings
from app.infrastructure.db.queries.statistics import (
    top_n_competition,
    bottom_n_competition,
    programs_with_free_places,
    applicant_with_most_applications,
    top_programs_by_avg_score, total_places, total_places_non_ino, total_applications, count_exam_submitted
)

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

session.close()
