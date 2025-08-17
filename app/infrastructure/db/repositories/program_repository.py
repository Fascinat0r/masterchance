# repositories/program_repository.py
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.orm import Session

from app.domain.models import (
    Institute, Department, Program,
    SubmissionStats, Applicant, Application, ProgramPassingQuantile, AdmissionProbability, AdmissionDiagnostics,
    ExamSession
)
from app.infrastructure.db.models import (
    InstituteModel, DepartmentModel, ProgramModel,
    SubmissionStatsModel, ApplicantModel, ApplicationModel, ProgramQuantileModel, AdmissionProbabilityModel,
    AdmissionDiagnosticsModel, ExamSessionModel
)


class ProgramRepository:
    def __init__(self, session: Session):
        self._session = session

    # ——— МАППЕРЫ ——————————————————————————————————————————————
    @staticmethod
    def _to_institute_model(inst: Institute) -> InstituteModel:
        return InstituteModel(code=inst.code, name=inst.name)

    @staticmethod
    def _to_department_model(dept: Department) -> DepartmentModel:
        return DepartmentModel(
            code=dept.code,
            name=dept.name,
            institute_code=dept.institute_code
        )

    @staticmethod
    def _to_program_model(prog: Program) -> ProgramModel:
        return ProgramModel(
            code=prog.code,
            name=prog.name,
            department_code=prog.department_code,
            is_ino=prog.is_ino,
            is_international=prog.is_international
        )

    @staticmethod
    def _to_institute_domain(model: InstituteModel) -> Institute:
        return Institute(code=model.code, name=model.name)

    @staticmethod
    def _to_department_domain(model: DepartmentModel) -> Department:
        return Department(
            code=model.code,
            name=model.name,
            institute_code=model.institute_code
        )

    @staticmethod
    def _to_program_domain(model: ProgramModel) -> Program:
        return Program(
            code=model.code,
            name=model.name,
            department_code=model.department_code,
            is_ino=model.is_ino,
            is_international=model.is_international
        )

    @staticmethod
    def _to_stats_model(s: SubmissionStats) -> SubmissionStatsModel:
        return SubmissionStatsModel(
            program_code=s.program_code,
            num_places=s.num_places,
            num_applications=s.num_applications,
            generated_at=s.generated_at
        )

    @staticmethod
    def _to_applicant_model(a: Applicant) -> ApplicantModel:
        return ApplicantModel(id=a.id)

    @staticmethod
    def _to_application_model(app: Application) -> ApplicationModel:
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

    @staticmethod
    def _to_stats_domain(m: SubmissionStatsModel) -> SubmissionStats:
        return SubmissionStats(
            program_code=m.program_code,
            num_places=m.num_places,
            num_applications=m.num_applications,
            generated_at=m.generated_at
        )

    @staticmethod
    def _to_applicant_domain(m: ApplicantModel) -> Applicant:
        return Applicant(id=m.id)

    @staticmethod
    def _to_application_domain(m: ApplicationModel) -> Application:
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

    @staticmethod
    def _to_quantile_model(q: ProgramPassingQuantile) -> ProgramQuantileModel:
        return ProgramQuantileModel(
            program_code=q.program_code,
            q90=q.q90,
            q95=q.q95,
        )

    @staticmethod
    def _to_probability_model(p: AdmissionProbability) -> AdmissionProbabilityModel:
        return AdmissionProbabilityModel(
            applicant_id=p.applicant_id,
            program_code=p.program_code,
            probability=p.probability,
        )

    @staticmethod
    def _to_exam_session_model(e: ExamSession) -> ExamSessionModel:
        return ExamSessionModel(
            program_code=e.program_code,
            exam_code=e.exam_code,
            dt=e.dt,
            institute=e.institute,
            education_form=e.education_form,
            contract=e.contract,
            program_name=e.program_name,
            program_pdf_url=e.program_pdf_url,
        )

    # ——— CRUD МЕТОДЫ ——————————————————————————————————————————————

    def clear_program_quantiles(self) -> None:
        self._session.execute(delete(ProgramQuantileModel))

    def clear_admission_probabilities(self) -> None:
        self._session.execute(delete(AdmissionProbabilityModel))

    def clear_admission_diagnostics(self) -> None:
        self._session.execute(delete(AdmissionDiagnosticsModel))

    def add_program_quantiles_bulk(self, quantiles: Iterable[ProgramPassingQuantile]) -> None:
        objs = [self._to_quantile_model(q) for q in quantiles]
        self._session.bulk_save_objects(objs)

    def add_admission_probabilities_bulk(self, probs: Iterable[AdmissionProbability]) -> None:
        objs = [self._to_probability_model(p) for p in probs]
        self._session.bulk_save_objects(objs)

    def add_admission_diagnostics_bulk(self, diags: Iterable[AdmissionDiagnostics]) -> None:
        objs = [self._to_diag_model(d) for d in diags]
        self._session.bulk_save_objects(objs)

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

    def get_diagnostics_for_applicant(self, applicant_id: str) -> AdmissionDiagnostics | None:
        r = (
            self._session.query(AdmissionDiagnosticsModel)
            .filter_by(applicant_id=applicant_id)
            .one_or_none()
        )
        if not r:
            return None
        return AdmissionDiagnostics(
            applicant_id=r.applicant_id,
            p_excluded=r.p_excluded,
            p_fail_when_included=r.p_fail_when_included,
        )

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

    def get_program_meta_df(self) -> "pd.DataFrame":
        """
        Вернуть DataFrame:
            program_code | department_code | is_international
        Используется Monte‑Carlo для построения exam_id.
        """
        rows = (
            self._session.query(
                ProgramModel.code,
                ProgramModel.department_code,
                ProgramModel.is_international,
            ).all()
        )
        return pd.DataFrame(
            rows, columns=["program_code", "department_code", "is_international"]
        )

    def get_latest_submission_generated_at(self) -> datetime | None:
        """
        Максимальная дата generated_at по всем программам (submission_stats).
        Возвращает datetime или None, если данных ещё нет.
        """
        ts = (
            self._session
            .query(func.max(SubmissionStatsModel.generated_at))
            .scalar()
        )
        return ts

    def clear_exam_sessions(self) -> None:
        self._session.execute(delete(ExamSessionModel))

    def add_exam_sessions_bulk(self, sessions: Iterable[ExamSession]) -> None:
        objs = [self._to_exam_session_model(s) for s in sessions]
        if objs:
            self._session.bulk_save_objects(objs)

    def get_exam_sessions_by_program(self, program_code: str) -> list[ExamSession]:
        rows = (
            self._session.query(ExamSessionModel)
            .filter_by(program_code=program_code)
            .order_by(ExamSessionModel.dt.asc())
            .all()
        )
        out: list[ExamSession] = []
        for r in rows:
            out.append(ExamSession(
                program_code=r.program_code,
                exam_code=r.exam_code,
                dt=r.dt,
                institute=r.institute,
                education_form=r.education_form,
                contract=r.contract,
                program_name=r.program_name,
                program_pdf_url=r.program_pdf_url,
            ))
        return out

    # ─── Поиск программы по имени в рамках кафедры ────────────────────────
    def find_program_code_by_name_and_dept(self, program_name: str, dept_code: str) -> str | None:
        """
        Возвращает код нашей программы по точному / почти точному совпадению имени
        в рамках заданной кафедры (department_code).
        """
        # 1) точное сравнение (регистр/пробелы игнорируем)
        q_exact = (
            self._session.query(ProgramModel.code)
            .filter(
                ProgramModel.department_code == dept_code,
                func.lower(func.trim(ProgramModel.name)) == func.lower(func.trim(program_name))
            )
        ).one_or_none()
        if q_exact:
            return q_exact[0]

        # 2) fallback: ILIKE по вхождению (на случай мелких расхождений)
        q_like = (
            self._session.query(ProgramModel.code)
            .filter(
                ProgramModel.department_code == dept_code,
                func.lower(ProgramModel.name).ilike(f"%{program_name.lower()}%")
            )
            .limit(2)
            .all()
        )
        if len(q_like) == 1:
            return q_like[0][0]
        return None

    def get_all_exam_sessions(self) -> list[ExamSession]:
        rows = (
            self._session.query(ExamSessionModel)
            .order_by(ExamSessionModel.program_code.asc(), ExamSessionModel.dt.asc())
            .all()
        )
        out: list[ExamSession] = []
        for r in rows:
            out.append(ExamSession(
                program_code=r.program_code,
                exam_code=r.exam_code,
                dt=r.dt,
                institute=r.institute,
                education_form=r.education_form,
                contract=r.contract,
                program_name=r.program_name,
                program_pdf_url=r.program_pdf_url,
            ))
        return out

    def commit(self) -> None:
        self._session.commit()
