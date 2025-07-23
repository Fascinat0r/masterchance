# repositories/program_repository.py

from sqlalchemy.orm import Session

from app.domain.models import (
    Institute, Department, Program,
    SubmissionStats, Applicant, Application
)
from app.infrastructure.db.models import (
    InstituteModel, DepartmentModel, ProgramModel,
    SubmissionStatsModel, ApplicantModel, ApplicationModel
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

    def commit(self) -> None:
        self._session.commit()
