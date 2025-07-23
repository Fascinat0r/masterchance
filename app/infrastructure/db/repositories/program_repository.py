# repositories/program_repository.py

from sqlalchemy.orm import Session

from app.domain.models import Institute, Department, Program
from app.infrastructure.db.models import InstituteModel, DepartmentModel, ProgramModel


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

    def commit(self) -> None:
        self._session.commit()
