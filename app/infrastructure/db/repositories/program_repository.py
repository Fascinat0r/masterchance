# repositories/program_repository.py
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.orm import Session

from app.domain.models import (
    Institute, Department, Program,
    SubmissionStats, Applicant, Application, ProgramPassingQuantile, AdmissionProbability
)
from app.infrastructure.db.models import (
    InstituteModel, DepartmentModel, ProgramModel,
    SubmissionStatsModel, ApplicantModel, ApplicationModel, ProgramQuantileModel, AdmissionProbabilityModel
)


class ProgramRepository:
    def __init__(self, session: Session):
        self._session = session

    # ——— МАППЕРЫ ——————————————————————————————————————————————
    def _to_institute_model(self, inst: Institute) -> InstituteModel:
        return InstituteModel(code=inst.code, name=inst.name)

    def _to_department_model(self, dept: Department) -> DepartmentModel:
        return DepartmentModel(
            code=dept.code,
            name=dept.name,
            institute_code=dept.institute_code
        )

    def _to_program_model(self, prog: Program) -> ProgramModel:
        return ProgramModel(
            code=prog.code,
            name=prog.name,
            department_code=prog.department_code,
            is_ino=prog.is_ino,
            is_international=prog.is_international
        )

    def _to_institute_domain(self, model: InstituteModel) -> Institute:
        return Institute(code=model.code, name=model.name)

    def _to_department_domain(self, model: DepartmentModel) -> Department:
        return Department(
            code=model.code,
            name=model.name,
            institute_code=model.institute_code
        )

    def _to_program_domain(self, model: ProgramModel) -> Program:
        return Program(
            code=model.code,
            name=model.name,
            department_code=model.department_code,
            is_ino=model.is_ino,
            is_international=model.is_international
        )

    def _to_stats_model(self, s: SubmissionStats) -> SubmissionStatsModel:
        return SubmissionStatsModel(
            program_code=s.program_code,
            num_places=s.num_places,
            num_applications=s.num_applications,
            generated_at=s.generated_at
        )

    def _to_applicant_model(self, a: Applicant) -> ApplicantModel:
        return ApplicantModel(id=a.id)

    def _to_application_model(self, app: Application) -> ApplicationModel:
        return ApplicationModel(
            program_code=app.program_code,
            applicant_id=app.applicant_id,
            total_score=app.total_score,
            vi_score=app.vi_score,
            subject1_score=app.subject1_score,
            subject2_score=app.subject2_score,
            id_achievements=app.id_achievements,
            target_id_achievements=app.target_id_achievements,
            priority=app.priority,
            consent=app.consent,
            review_status=app.review_status
        )

    def _to_stats_domain(self, m: SubmissionStatsModel) -> SubmissionStats:
        return SubmissionStats(
            program_code=m.program_code,
            num_places=m.num_places,
            num_applications=m.num_applications,
            generated_at=m.generated_at
        )

    def _to_applicant_domain(self, m: ApplicantModel) -> Applicant:
        return Applicant(id=m.id)

    def _to_application_domain(self, m: ApplicationModel) -> Application:
        return Application(
            program_code=m.program_code,
            applicant_id=m.applicant_id,
            total_score=m.total_score,
            vi_score=m.vi_score,
            subject1_score=m.subject1_score,
            subject2_score=m.subject2_score,
            id_achievements=m.id_achievements,
            target_id_achievements=m.target_id_achievements,
            priority=m.priority,
            consent=m.consent,
            review_status=m.review_status
        )

    def _to_quantile_model(self, q: ProgramPassingQuantile) -> ProgramQuantileModel:
        return ProgramQuantileModel(
            program_code=q.program_code,
            q90=q.q90,
            q95=q.q95,
        )

    def _to_probability_model(self, p: AdmissionProbability) -> AdmissionProbabilityModel:
        return AdmissionProbabilityModel(
            applicant_id=p.applicant_id,
            program_code=p.program_code,
            probability=p.probability,
        )

    # ——— CRUD МЕТОДЫ ——————————————————————————————————————————————
    def add_institute(self, inst: Institute) -> None:
        orm = self._to_institute_model(inst)
        self._session.merge(orm)

    def add_department(self, dept: Department) -> None:
        # Гарантируем, что институт существует
        # (если нет — предварительно вызвать add_institute)
        orm = self._to_department_model(dept)
        self._session.merge(orm)

    def add_program(self, prog: Program) -> None:
        # Гарантируем, что кафедра существует
        # (если нет — предварительно вызвать add_department)
        orm = self._to_program_model(prog)
        self._session.merge(orm)

    def get_all_institutes(self) -> list[Institute]:
        models = self._session.query(InstituteModel).all()
        return [self._to_institute_domain(m) for m in models]

    def get_departments_by_institute(self, institute_code: str) -> list[Department]:
        models = (
            self._session
            .query(DepartmentModel)
            .filter_by(institute_code=institute_code)
            .all()
        )
        return [self._to_department_domain(m) for m in models]

    def get_programs_by_department(self, department_code: str) -> list[Program]:
        models = (
            self._session
            .query(ProgramModel)
            .filter_by(department_code=department_code)
            .all()
        )
        return [self._to_program_domain(m) for m in models]

    def add_submission_stats(self, stats: SubmissionStats) -> None:
        orm = self._to_stats_model(stats)
        self._session.merge(orm)

    def add_applicant(self, applicant: Applicant) -> None:
        orm = self._to_applicant_model(applicant)
        self._session.merge(orm)

    def add_application(self, application: Application) -> None:
        # гарантируем, что абитуриент и направление есть
        self.add_applicant(Applicant(id=application.applicant_id))
        orm = self._to_application_model(application)
        self._session.merge(orm)

    def get_submission_stats(self, program_code: str) -> SubmissionStats | None:
        m = self._session.query(SubmissionStatsModel) \
            .filter_by(program_code=program_code) \
            .one_or_none()
        return self._to_stats_domain(m) if m else None

    def get_applications_by_program(self, program_code: str) -> list[Application]:
        ms = self._session.query(ApplicationModel) \
            .filter_by(program_code=program_code) \
            .all()
        return [self._to_application_domain(m) for m in ms]

    def get_applicants_department_avg_subject1(self):
        """
        Вернуть список кортежей: (applicant_id, department_code, avg_subject1_score)
        Если у одного студента по одной кафедре несколько заявок — усреднять баллы.
        Только для заявок с ненулевым баллом за экзамен.
        """
        q = (
            self._session.query(
                ApplicationModel.applicant_id,
                ProgramModel.department_code,
                func.avg(ApplicationModel.subject1_score).label("avg_subject1_score")
            )
            .join(ProgramModel, ApplicationModel.program_code == ProgramModel.code)
            .filter(ApplicationModel.subject1_score > 0)
            .filter(ApplicationModel.subject1_score != 100)
            .group_by(ApplicationModel.applicant_id, ProgramModel.department_code)
        )
        # Вернуть как список кортежей
        return q.all()

    def get_all_applications(self) -> list[Application]:
        models = self._session.query(ApplicationModel).all()
        return [self._to_application_domain(m) for m in models]

    def get_all_applicants(self) -> list[Applicant]:
        models = self._session.query(ApplicantModel).all()
        return [self._to_applicant_domain(m) for m in models]

    def get_all_submission_stats(self) -> list[SubmissionStats]:
        models = self._session.query(SubmissionStatsModel).all()
        return [self._to_stats_domain(m) for m in models]

    # ────────── Monte‑Carlo CRUD ──────────────────────────────────────────
    # очистка
    def clear_program_quantiles(self) -> None:
        self._session.execute(delete(ProgramQuantileModel))

    def clear_admission_probabilities(self) -> None:
        self._session.execute(delete(AdmissionProbabilityModel))

    # массовые вставки
    def add_program_quantiles_bulk(self, quantiles: Iterable[ProgramPassingQuantile]) -> None:
        objs = [self._to_quantile_model(q) for q in quantiles]
        self._session.bulk_save_objects(objs)

    def add_admission_probabilities_bulk(self, probs: Iterable[AdmissionProbability]) -> None:
        objs = [self._to_probability_model(p) for p in probs]
        self._session.bulk_save_objects(objs)

    # чтение
    def get_probabilities_for_applicant(self, applicant_id: str) -> List[AdmissionProbability]:
        rows = (
            self._session.query(AdmissionProbabilityModel)
            .filter_by(applicant_id=applicant_id)
            .all()
        )
        return [
            AdmissionProbability(
                applicant_id=r.applicant_id,
                program_code=r.program_code,
                probability=r.probability,
            )
            for r in rows
        ]

    def get_quantiles_for_programs(self, codes: Iterable[str]) -> Dict[str, ProgramPassingQuantile]:
        result: Dict[str, ProgramPassingQuantile] = {}
        rows = (
            self._session.query(ProgramQuantileModel)
            .filter(ProgramQuantileModel.program_code.in_(list(codes)))
            .all()
        )
        for r in rows:
            result[r.program_code] = ProgramPassingQuantile(
                program_code=r.program_code,
                q90=r.q90,
                q95=r.q95,
            )
        return result

    def get_programs_by_codes(self, codes: Sequence[str]) -> Dict[str, Program]:
        """
        Вернуть {program_code: Program} только для требуемых записей.
        """
        if not codes:
            return {}
        rows = (
            self._session.query(ProgramModel)
            .filter(ProgramModel.code.in_(list(codes)))
            .all()
        )
        return {m.code: self._to_program_domain(m) for m in rows}

    # ---- чтение кафедр по списку кодов -------------------------------------
    def get_departments_by_codes(self, codes: Sequence[str]) -> Dict[str, Department]:
        """
        {dept_code: Department}
        """
        if not codes:
            return {}
        rows = (
            self._session.query(DepartmentModel)
            .filter(DepartmentModel.code.in_(list(codes)))
            .all()
        )
        return {m.code: self._to_department_domain(m) for m in rows}

    def get_program_codes_by_applicant(self, applicant_id: str) -> List[str]:
        """
        Вернуть *уникальный* список program_code, на которые подана заявка
        данным applicant_id (порядок по минимальному приоритету).
        """
        rows = (
            self._session.query(
                ApplicationModel.program_code,
                func.min(ApplicationModel.priority).label("min_prio")
            )
            .filter(ApplicationModel.applicant_id == applicant_id)
            .group_by(ApplicationModel.program_code)
            .order_by("min_prio")
            .all()
        )
        return [r.program_code for r in rows]

    def delete_applications_by_program(self, program_code: str) -> None:
        """
        Жёстко удалить все ApplicationModel, привязанные к program_code.
        """
        self._session.execute(
            delete(ApplicationModel).where(ApplicationModel.program_code == program_code)
        )

    def add_applicants_bulk(self, applicant_ids: Iterable[str]) -> None:
        """
        Массовое добавление / UPSERT абитуриентов.
        Работает быстро и без повторов.
        """
        if not applicant_ids:
            return

        stmt = sqlite_upsert(ApplicantModel).values(
            [{"id": aid} for aid in set(applicant_ids)]
        ).on_conflict_do_nothing(index_elements=["id"])
        self._session.execute(stmt)

    def add_applications_bulk(self, applications: Iterable[Application]) -> None:
        """
        Массовая вставка / upsert заявок.
        """
        apps = list(applications)
        if not apps:
            return

        # превращаем в «сырые» словари (уск. bulk)
        rows: list[dict] = []
        for a in apps:
            model = self._to_application_model(a)
            d = {col.name: getattr(model, col.name)
                 for col in ApplicationModel.__table__.columns}
            rows.append(d)

        # 1. базовый INSERT
        insert_stmt = sqlite_insert(ApplicationModel).values(rows)

        # 2. ON CONFLICT DO UPDATE (по composite PK)
        update_cols = {
            "total_score": insert_stmt.excluded.total_score,
            "vi_score": insert_stmt.excluded.vi_score,
            "subject1_score": insert_stmt.excluded.subject1_score,
            "subject2_score": insert_stmt.excluded.subject2_score,
            "id_achievements": insert_stmt.excluded.id_achievements,
            "target_id_achievements": insert_stmt.excluded.target_id_achievements,
            "priority": insert_stmt.excluded.priority,
            "consent": insert_stmt.excluded.consent,
            "review_status": insert_stmt.excluded.review_status,
        }

        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["program_code", "applicant_id"],
            set_=update_cols,
        )
        self._session.execute(upsert_stmt)

    def commit(self) -> None:
        self._session.commit()
