#!/usr/bin/env python3
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.models import Institute, Department, Program
from app.infrastructure.db.models import Base
from app.infrastructure.db.repositories.program_repository import ProgramRepository

def main():
    # 1) Настройка БД
    engine = create_engine('sqlite:///master.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    repo = ProgramRepository(session)

    # 2) Загрузка JSON с ключом "data"
    with open("directions_master.json", encoding="utf-8") as json_file:
        payload = json.load(json_file)
        items = payload.get("data", [])

    # 3) Преобразование каждого элемента в доменные модели
    for item in items:
        full_name = item["name"]
        # Разбиваем: '01.04.02 Название кафедры (Название программы) (ИНО?)'
        dept_code, remainder = full_name.split(" ", 1)
        dept_name, paren = remainder.split("(", 1)
        prog_name = paren.rsplit(")", 1)[0]

        is_ino = full_name.endswith("(ИНО)")
        is_intl = "международная образовательная программа" in full_name.lower()

        inst_code = dept_code.split(".")[0]
        institute = Institute(code=inst_code, name="")  # Название института можно задать вручную
        department = Department(
            code=dept_code,
            name=dept_name.strip(),
            institute_code=inst_code
        )
        program = Program(
            code=item["code"],
            name=prog_name.strip(),
            department_code=dept_code,
            is_ino=is_ino,
            is_international=is_intl
        )

        repo.add_institute(institute)
        repo.add_department(department)
        repo.add_program(program)

    # 4) Сохраняем всё в БД
    repo.commit()
    print(f"Импортировано {len(items)} направлений.")

if __name__ == "__main__":
    main()
