from dataclasses import dataclass


@dataclass
class Institute:
    """
    Институт: определяется по первым двум цифрам кода кафедры.
    Например, код кафедры '01.04.02' → institute_code='01'.
    """
    code: str  # '01'
    name: str  # название института


@dataclass
class Department:
    """
    Кафедра (department): код всей тройки цифр, напр. '01.04.02',
    name — текст между кодом и первой скобкой.
    """
    code: str  # '01.04.02'
    name: str  # 'Прикладная математика и информатика'
    institute_code: str  # ссылка на Institute.code


@dataclass
class Program:
    """
    Направление обучения (program):
    - code: внутренний числовой код, напр. '786'
    - name: текст в первой паре скобок
    - is_ino: пометка «(ИНО)»
    - is_international: содержит «международная образовательная программа»
    - department_code: ссылка на Department.code
    """
    code: str
    name: str
    department_code: str
    is_ino: bool = False
    is_international: bool = False
