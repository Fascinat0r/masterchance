"""
Telegram-бот, показывающий направления, квантили и шансы.
"""
import asyncio
from datetime import datetime, timedelta
from textwrap import dedent
from typing import List, Dict
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.application.use_cases.get_last_update_time import GetLastUpdateTimeUseCase
from app.config.config import settings
from app.config.logger import logger
from app.domain.models import Program, Application, ExamSession
from app.infrastructure.db.models import Base
from app.infrastructure.db.repositories.program_repository import ProgramRepository

_engine = create_engine(settings.database_url, echo=False, future=True)
Base.metadata.create_all(_engine)
_Session = sessionmaker(bind=_engine, future=True)

_SRC_TZ = ZoneInfo("Europe/Moscow")  # расписание и сайт — МСК


def split_message(text: str, max_len: int = 4000) -> List[str]:
    parts = []
    while len(text) > max_len:
        split_idx = text.rfind('\n', 0, max_len)
        if split_idx == -1:
            split_idx = max_len
        parts.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        parts.append(text)
    return parts


def _fmt_qrange(q) -> str:
    """
    Преобразует квантили в строку `X - X`. Если нет данных — вернёт '—'.
    """
    if not q:
        return "—"
    try:
        if q.q90 == q.q95:
            return f"{q.q90:.0f}"
        else:
            return f"{q.q90:.0f} - {q.q95:.0f}"
    except Exception:
        return "—"


def _human_prog_line(dept_code: str, prog_name: str) -> str:
    return f"• `{dept_code}`  *{prog_name}*"


def _fmt_local_from_msk_naive(dt_naive_msk: datetime) -> str:
    """
    На входе tz-naive время в МСК (как в БД). Возвращаем строку в целевой зоне settings.timezone.
    """
    aware_msk = dt_naive_msk.replace(tzinfo=_SRC_TZ)
    local = aware_msk.astimezone(settings.timezone)
    # без года — компактнее, но при желании можно добавить %Y
    return local.strftime("%d.%m %H:%M")


def _exam_info_line(app: Application | None, sessions: List[ExamSession] | None) -> str | None:
    """
    Возвращает одну «подстрочку» для направления:
      • если есть баллы — показываем их (без второго предмета);
      • иначе — ближайшие даты экзаменов (до 3 шт.);
      • если дат нет — сообщаем, что расписания пока нет / экзамены прошли.
      • если последний экзамен был < 3 дней назад — добавляем предупреждение.
    """
    # 1) Есть результат
    if app and ((app.vi_score and app.vi_score > 0) or (app.subject1_score and app.subject1_score > 0)):
        parts = []
        if app.vi_score > 0:
            parts.append(f"{app.vi_score}")
        if app.id_achievements > 0:
            parts.append(f"+{app.id_achievements}")
        if app.target_id_achievements > 0:
            parts.append(f"+{app.target_id_achievements}")
        parts.append(f"=**{app.total_score}**")
        detail = "".join(parts) if parts else "баллы получены"
        return f"   ↳ 🟢 Cдан: {detail}"

    # 2) Нет результата — подскажем даты
    sessions = sessions or []
    if not sessions:
        return "   ↳ 🟡 Расписание экзамена пока не опубликовано"

    now_msk = datetime.now(_SRC_TZ)
    upcoming = [s for s in sessions if s.dt.replace(tzinfo=_SRC_TZ) >= now_msk]
    if upcoming:
        show = upcoming[:3]
        dates = "; ".join(_fmt_local_from_msk_naive(s.dt) for s in show)
        more = " …" if len(upcoming) > 3 else ""
        return f"   ↳ 🟡 Ближайшие экзамены: {dates}{more}"
    else:
        last_dt = sessions[-1].dt
        line = f"   ↳ ⚪ Экзамены завершились (последняя дата: {_fmt_local_from_msk_naive(last_dt)})"
        # Предупреждение: прошло < 3 дней после последнего экзамена
        try:
            last_aware = last_dt.replace(tzinfo=_SRC_TZ)
            delta = now_msk - last_aware
            if delta.total_seconds() >= 0 and delta < timedelta(days=3):
                line += "\n   ↳ ⚠️ прошло < 3 дней — результаты могут ещё обновляться."
        except Exception:
            pass
        return line


def _format_response(applicant_id: str,
                     all_codes: List[str],
                     probs_uncond: Dict[str, float],
                     quantiles,
                     prog_map: Dict[str, Program],
                     diag,
                     apps_by_code: Dict[str, Application],
                     sessions_by_code: Dict[str, List[ExamSession]]) -> str:
    """
    Новый порядок:
      1) Список направлений + строка про экзамены (баллы/даты).
      2) 🔮 Прогноз зачисления (условные вероятности), под каждой строкой — «проходной: X - X».
      3) Блок про «пролёт».
    """
    if not all_codes:
        return f"У абитуриента `{applicant_id}` нет поданных заявок 🤷‍♂️"

    # 1) Направления + экзамены
    head_programs = "📝 *Ваши направления*"
    prog_lines: List[str] = []
    for code in all_codes:
        prog = prog_map.get(code)
        line = _human_prog_line(
            dept_code=(prog.department_code if prog else code.split('.')[0]),
            prog_name=(prog.name if prog else code),
        )
        prog_lines.append(line)

        app = apps_by_code.get(code)
        sess = sessions_by_code.get(code, [])
        exam_line = _exam_info_line(app, sess)
        if exam_line:
            prog_lines.append(exam_line)

    # 2) Вероятности (условные) + проходные ниже той же строки
    p_excl = diag.p_excluded if diag else 0.0
    p_incl = max(1.0 - p_excl, 1e-9)
    probs_cond: Dict[str, float] = {k: min(v / p_incl, 1.0) for k, v in probs_uncond.items()}

    head_forecast = "\n\n🔮 *Прогноз зачисления*"
    forecast_lines: List[str] = []
    for code in all_codes:
        pname = prog_map[code].name if code in prog_map else code
        p = probs_cond.get(code)
        p_str = f"{p * 100:.1f}%" if p is not None else "—"
        q = quantiles.get(code)
        forecast_lines.append(f"• `{pname}`  →  *{p_str}* (проходной: {_fmt_qrange(q)})")

    # 3) «Пролёт»
    fail_uncond = max(0.0, 1.0 - sum(probs_uncond.values()))
    fail_cond = min(1.0, (diag.p_fail_when_included if diag else fail_uncond / p_incl))
    head_fail = (
        "\n\n🚫 *«Пролетел с магой»*\n"
        f"• В *{fail_cond * 100:.1f}%* симуляций\n"
    )

    return "\n".join([head_programs, *prog_lines, head_forecast, *forecast_lines, head_fail])


async def how_cmd(msg: Message):
    await msg.answer(dedent("""
    🧠 *Как работает прогноз?*
    
    Прогноз построен по методу Монте‑Карло — это способ смоделировать тысячи возможных будущих сценариев. Вот как это работает шаг за шагом:

    1. **Повторяем симуляцию десятки тысяч раз** — это как доктор Стрэндж, просматривающий альтернативные вселенные.

    2. **У кого уже есть балл по вступительному — он сохраняется.**

    3. **Если балла по экзамену нет**, он симулируется (считается случайно), но не просто так:
       • если ты уже сдавал хотя бы на одном направлении — считаем что в среднем ты сдашь примерно также и остальные.
       • если ещё не сдавал, но есть заявки на направления, где другие уже сдавали — берём статистику по ним;
       • если ни ты, ни другие ничего не сдавали — используем глобальную статистику по всем.

    4. **После этого балл складывается с индивидуальными достижениями**, и получается полный конкурсный балл.

    5. **Имитация конкурса как в вузе**:
       • сначала все абитуриенты «распределяются» по 1‑м приоритетам;
       • если кто-то не проходит — перекидываются на 2‑й приоритет, и так далее.

    6. **Результат:**
       • если ты попал на направление в 8 000 симуляциях из 10 000, шанс ≈ 80%;
       • считаем также «средний» и «высокий» проходной балл (90 % и 95 % квантиль).

    ⚠️ *Предсказания не гарантируют поступление!*
    Это всего лишь вероятностная модель на основе того, что уже известно.
    **Также, модель не учитывает:**
        • Наличие заявок в другие вузы (Невозможно достать такую информацию).
        • "Неявку" на вступительные испытания (Невозможно понять, намерен ли человек идти на какой либо из своих экзаменов, даже если он подавал заявку).
    В следствии чего шансы принято считать пессимистичными (заниженными).
    
    """).strip(), parse_mode="Markdown")


async def start_cmd(msg: Message):
    session = _Session()
    repo = ProgramRepository(session)
    try:
        last_dt = GetLastUpdateTimeUseCase(repo).execute()
    except Exception:
        last_dt = None
    finally:
        session.close()

    def _fmt(dt: datetime | None) -> str:
        return dt.strftime("%d.%m.%Y %H:%M") if dt else "нет данных"

    await msg.answer(
        dedent(f"""
        Привет! Отправь мне **код абитуриента** — покажу все направления, шанс зачисления и ориентиры проходных баллов.

        Последнее обновление данных: **{_fmt(last_dt)}**

        /how — как это работает?
        По вопросам писать @fascinat00r
        """).strip(),
        parse_mode="Markdown"
    )


async def applicant_handler(msg: Message):
    applicant_id = msg.text.strip()
    if not applicant_id:
        return

    session = _Session()
    repo = ProgramRepository(session)

    try:
        all_codes = repo.get_program_codes_by_applicant(applicant_id)
        if not all_codes:
            await msg.answer(
                f"Не найдено заявок для абитуриента `{applicant_id}`.",
                parse_mode="Markdown"
            )
            return

        # Вероятности / квантили / метаданные программ
        prob_objs = repo.get_probabilities_for_applicant(applicant_id)
        probs_uncond = {p.program_code: p.probability for p in prob_objs}

        quantiles = repo.get_quantiles_for_programs(all_codes)
        prog_map = repo.get_programs_by_codes(all_codes)

        diag = repo.get_diagnostics_for_applicant(applicant_id)

        # Новое: заявки абитуриента с баллами по конкретным программам
        apps = repo.get_applications_by_applicant(applicant_id)
        apps_by_code: Dict[str, Application] = {a.program_code: a for a in apps if a.program_code in all_codes}

        # Новое: ближайшие экзамены по каждой программе
        sessions_by_code: Dict[str, List[ExamSession]] = {}
        for code in all_codes:
            sessions_by_code[code] = repo.get_exam_sessions_by_program(code)

        full_text = _format_response(
            applicant_id, all_codes, probs_uncond, quantiles, prog_map, diag,
            apps_by_code=apps_by_code, sessions_by_code=sessions_by_code
        )

        for part in split_message(full_text):
            try:
                await msg.answer(part, parse_mode="Markdown")
            except TelegramBadRequest:
                await msg.answer("⚠️ Не удалось отправить сообщение (возможно, проблема с Markdown).")
                break

    except Exception as exc:
        logger.exception("TG-handler error: %s", exc)
        await msg.answer("Произошла ошибка 😥")
    finally:
        session.close()


def start_bot(bot_token) -> None:
    bot = Bot(bot_token)
    dp = Dispatcher()

    dp.message.register(start_cmd, CommandStart())
    dp.message.register(how_cmd, Command("how"))
    dp.message.register(applicant_handler, F.text)

    logger.info("Telegram-бот запущен.")
    asyncio.run(dp.start_polling(bot))
