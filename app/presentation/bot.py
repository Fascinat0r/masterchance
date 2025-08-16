"""
Telegram-бот, показывающий направления, квантили и шансы.
"""
import asyncio
from datetime import datetime
from textwrap import dedent
from typing import List, Dict

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart, Command, CommandObject
from aiogram.types import Message
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.application.use_cases.get_last_update_time import GetLastUpdateTimeUseCase
from app.config.config import settings
from app.config.logger import logger
from app.domain.models import Program
from app.infrastructure.db.models import Base
from app.infrastructure.db.repositories.program_repository import ProgramRepository

_engine = create_engine(settings.database_url, echo=False, future=True)
Base.metadata.create_all(_engine)
_Session = sessionmaker(bind=_engine, future=True)


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


def _human_prog_line(dept_code: str, prog_name: str, q90: float | None, q95: float | None) -> str:
    safe = f"{q90:.0f}" if q90 is not None else "—"
    high = f"{q95:.0f}" if q95 is not None else "—"
    return f"• `{dept_code}`  *{prog_name}*  —  средний ={safe}, высокий ={high}"


def _format_response(applicant_id: str,
                     all_codes: List[str],
                     probs_uncond: Dict[str, float],
                     quantiles,
                     prog_map: Dict[str, Program],
                     diag) -> str:
    """
    Формирует Markdown-ответ. Показывает:
      1) Квантили;
      2) Вероятности «если вы точно идёте в Политех» (условные);
      3) «С учётом оттока 20%» (безусловные);
      4) Во сколько % симуляций вы «пролетели с магой» (оба варианта).
    """
    if not all_codes:
        return f"У абитуриента `{applicant_id}` нет поданных заявок 🤷‍♂️"

    head1 = "📝 *Ваши направления и ориентиры балла*"
    prog_lines: List[str] = []
    for code in all_codes:
        prog = prog_map.get(code)
        q = quantiles.get(code)
        prog_lines.append(
            _human_prog_line(
                dept_code=(prog.department_code if prog else code.split('.')[0]),
                prog_name=(prog.name if prog else code),
                q90=(q.q90 if q else None),
                q95=(q.q95 if q else None),
            )
        )

    # Диагностика
    p_excl = diag.p_excluded if diag else 0.0
    p_incl = max(1.0 - p_excl, 1e-9)

    # Условные вероятности (как если пользователь точно остаётся)
    probs_cond: Dict[str, float] = {k: min(v / p_incl, 1.0) for k, v in probs_uncond.items()}

    # «Пролетел»:
    fail_uncond = max(0.0, 1.0 - sum(probs_uncond.values()))
    fail_cond   = min(1.0, (diag.p_fail_when_included if diag else fail_uncond / p_incl))

    # Блоки вероятностей
    head2 = "\n\n🔮 *Вероятность зачисления*"
    lines_cond: List[str] = []
    for code in all_codes:
        p = probs_cond.get(code)
        if p is not None:
            pname = prog_map[code].name if code in prog_map else code
            lines_cond.append(f"• `{pname}`  →  *{p * 100:.1f}%*")

    # head3 = "\n\n♻️ *С учётом 20% оттока (общая модель)*"
    # lines_uncond: List[str] = []
    # for code in all_codes:
    #     p = probs_uncond.get(code)
    #     if p is not None:
    #         pname = prog_map[code].name if code in prog_map else code
    #         lines_uncond.append(f"• `{pname}`  →  *{p * 100:.1f}%*")

    head4 = (
        "\n\n🚫 *«Пролетел с магой»*\n"
        f"• В *{fail_cond*100:.1f}%* симуляций\n"
    )

    # По умолчанию показываем условные сверху (настраивается в settings)
    if settings.bot_show_anchored:
        return "\n".join([head1, *prog_lines, head2, *lines_cond, head4])
    else:
        return "\n".join([head1, *prog_lines, head4, head2, *lines_cond])


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
        • Наличие заявок в другие вузы.
        • "Неявку" на вступительные испытания.
    В следствии чего шансы принято считать пессимистичными.
    
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
        Привет! Отправь мне **код абитуриента** — покажу все направления, 
        куда поданы документы, «средний» (90 %) и «высокий» (95 %) 
        проходные баллы и шанс зачисления.

        Последнее обновление данных: **{_fmt(last_dt)}**

        /how — как это работает?
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

        prob_objs = repo.get_probabilities_for_applicant(applicant_id)
        probs_uncond = {p.program_code: p.probability for p in prob_objs}

        quantiles = repo.get_quantiles_for_programs(all_codes)
        prog_map = repo.get_programs_by_codes(all_codes)

        diag = repo.get_diagnostics_for_applicant(applicant_id)

        full_text = _format_response(applicant_id, all_codes, probs_uncond, quantiles, prog_map, diag)
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
