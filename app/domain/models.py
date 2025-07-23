import datetime
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


@dataclass
class SubmissionStats:
    """
    Сводка по направлению.
    """
    program_code: str  # FK → Program.code
    num_places: int  # из <span id="numPlaces">
    num_applications: int  # из <span id="numApplications">
    generated_at: datetime  # время формирования из <span id="date_info">


@dataclass
class Applicant:
    """
    Абитуриент.
    """
    id: str  # Уникальный код поступающего


@dataclass
class Application:
    """
    Заявка абитуриента по направлению.
    """
    program_code: str  # FK → Program.code
    applicant_id: str  # FK → Applicant.id
    total_score: int  # Сумма конкурсных баллов
    vi_score: int  # Сумма баллов за ВИ
    subject1_score: int  # Баллы по предмету 1
    subject2_score: int  # Баллы по предмету 2
    id_achievements: int  # Баллы за общие ИД (индивидуальные достижения)
    target_id_achievements: int  # Баллы за целевые ИД
    priority: int  # Приоритет
    consent: bool  # Наличие согласия (“+” → True)
    review_status: str  # Информация о рассмотрении
