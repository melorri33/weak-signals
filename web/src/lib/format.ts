/**
 * Подписи и форматирование — всё, что пользователь читает, по-русски.
 * Пороги и шаги совпадают с бэкендом: CONFIDENT_THRESHOLD в src/common/schemas.py,
 * шаги — вызовы progress() в src/pipeline/run.py.
 */
import type {
  ReasonCode,
  SignalCard,
  SourceRef,
  SourceType,
  TrustLevel,
} from "@/api/client"

export const TOP_N = 15
export const CONFIDENT_THRESHOLD = 0.75
const MID_THRESHOLD = 0.5

export type Level = "high" | "mid" | "low"

export function confidenceLevel(score: number): Level {
  if (score > CONFIDENT_THRESHOLD) return "high"
  if (score >= MID_THRESHOLD) return "mid"
  return "low"
}

export const CONFIDENCE_LABEL: Record<Level, string> = {
  high: "Высокая",
  mid: "Средняя",
  low: "Низкая",
}

/** Тот же смысл, что confidence_text в Streamlit-интерфейсе, — жюри не увидит двух разных толкований. */
export const CONFIDENCE_TEXT: Record<Level, string> = {
  high: "Модель уверена, что это ранняя технология",
  mid: "Скорее ранняя технология, но уверенности мало",
  low: "Слабая оценка: в выдаче, потому что топ-15 заполняется целиком",
}

export function percent(score: number): string {
  return `${Math.round(score * 100)}%`
}

export const TRUST_LEVEL: Record<TrustLevel, Level> = {
  high: "high",
  medium: "mid",
  low: "low",
}

export const TRUST_LABEL: Record<TrustLevel, string> = {
  high: "Высокое доверие",
  medium: "Среднее доверие",
  low: "Низкое доверие",
}

/** Как ТЗ определяет уровни: жюри спросит «по какому критерию». */
export const TRUST_CRITERIA: Record<TrustLevel, string> = {
  high: "Научные журналы, патентные базы, госорганы и регуляторы, университеты",
  medium: "Отраслевые СМИ, аналитические отчёты, препринты",
  low: "Блоги, соцсети, агрегаторы, пресс-релизы — только как подсказка",
}

export const SOURCE_TYPE_LABEL: Record<SourceType, string> = {
  paper: "Научная статья",
  preprint: "Препринт",
  patent: "Патент",
  news: "Новость",
  report: "Аналитический отчёт",
  gov: "Госорган или регулятор",
  blog: "Блог",
  social: "Соцсеть",
  press_release: "Пресс-релиз",
  other: "Другое",
}

const LANGUAGE_LABEL: Record<string, string> = {
  ru: "Русский",
  en: "Английский",
  zh: "Китайский",
  de: "Немецкий",
  fr: "Французский",
  ja: "Японский",
}

export function languageLabel(code: string): string {
  return LANGUAGE_LABEL[code.toLowerCase()] ?? code.toUpperCase()
}

export const REASON_LABEL: Record<ReasonCode, string> = {
  mature: "Зрелая технология",
  hype: "Медийная тема",
  noise: "Недостаточно подтверждений",
  no_research: "Нет следа в исследованиях",
  ok: "Оставлено",
}

/** Что правило значит простыми словами — для заголовков групп отсева. */
export const REASON_MEANING: Record<ReasonCode, string> = {
  mature:
    "Массовые технологии со сформированным рынком: публикаций тысячи, есть стандарты",
  hype: "О технологии много пишут в СМИ, но мало исследуют — признак маркетингового шума",
  noise:
    "Все источники — блоги, соцсети или пресс-релизы, независимого подтверждения нет",
  no_research:
    "Ни одной научной работы по точному названию — обычно так выглядит название продукта",
  ok: "Правила отсева не сработали",
}

export const REASON_ORDER: ReasonCode[] = [
  "mature",
  "hype",
  "noise",
  "no_research",
  "ok",
]

/** Шаги конвейера в порядке выполнения: stage из SearchResult → понятное объяснение. */
export const PIPELINE_STAGES: {
  stage: string
  title: string
  detail: string
}[] = [
  {
    stage: "расширяем запрос",
    title: "Формулируем поисковые фразы",
    detail:
      "Языковая модель превращает запрос в 16–22 фразы, в основном на английском",
  },
  {
    stage: "собираем источники",
    title: "Собираем источники",
    detail:
      "Техноновости, arXiv и OpenAlex — параллельно, у каждого источника свой таймаут",
  },
  {
    stage: "выделяем кандидатов",
    title: "Выделяем технологии",
    detail: "Модель читает документы и выписывает узкие названия технологий",
  },
  {
    stage: "считаем признаки",
    title: "Считаем признаки",
    detail:
      "Динамика публикаций, доля препринтов, упоминания в медиа, статья в Википедии",
  },
  {
    stage: "отсев и скоринг",
    title: "Отсеиваем и оцениваем",
    detail:
      "Правила убирают зрелое, хайп и шум, классификатор оценивает остальное",
  },
  {
    stage: "ищем документы по кандидатам",
    title: "Ищем подтверждения",
    detail: "Второй круг поиска по точному названию каждого кандидата",
  },
  {
    stage: "собираем карточки",
    title: "Пишем карточки",
    detail: "Описание, преимущество и кейс — только по найденным документам",
  },
]

export function stageIndex(stage: string): number {
  return PIPELINE_STAGES.findIndex((s) => s.stage === stage)
}

const dateFormat = new Intl.DateTimeFormat("ru-RU", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
})
const dateTimeFormat = new Intl.DateTimeFormat("ru-RU", {
  day: "numeric",
  month: "long",
  hour: "2-digit",
  minute: "2-digit",
})
const numberFormat = new Intl.NumberFormat("ru-RU")

export function formatDate(value: string | null | undefined): string {
  if (!value) return "Дата не указана"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : dateFormat.format(date)
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "Время не записано"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : dateTimeFormat.format(date)
}

export function formatNumber(value: number): string {
  return numberFormat.format(value)
}

export function formatDuration(seconds: number): string {
  const total = Math.round(seconds)
  const min = Math.floor(total / 60)
  const sec = total % 60
  return min > 0 ? `${min} мин ${sec} с` : `${sec} с`
}

/** ТЗ: соцсети, блоги, агрегаторы и пресс-релизы не могут быть единственным основанием. */
export function hasOnlyLowTrust(card: SignalCard): boolean {
  return card.sources.length > 0 && card.sources.every((s) => s.trust === "low")
}

/** Отчёты и документы регуляторов — раздел «Оценки в аналитических отчётах». */
export function analyticalSources(sources: SourceRef[]): SourceRef[] {
  return sources.filter(
    (s) => s.source_type === "report" || s.source_type === "gov"
  )
}

export function displayName(card: {
  name: string
  name_ru?: string | null
}): string {
  return card.name_ru?.trim() || card.name
}

/** Склонение: 1 источник, 2 источника, 5 источников. */
export function plural(
  n: number,
  one: string,
  few: string,
  many: string
): string {
  const mod10 = n % 10
  const mod100 = n % 100
  if (mod10 === 1 && mod100 !== 11) return one
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few
  return many
}

/**
 * Причина отказа источника без сырого URL: «406 Not Acceptable», «таймаут».
 * Полный текст остаётся в подсказке и в JSON прогона.
 */
export function failureReason(detail: string): string {
  const status =
    /'(\d{3} [^']+)'/.exec(detail) ?? /\b(\d{3} [A-Z][A-Za-z ]+)/.exec(detail)
  if (status) return status[1].trim()
  if (/timeout/i.test(detail)) return "не ответил за отведённое время"
  const short = detail
    .split(/ for url /i)[0]
    .replace(/^\w+Error:\s*/, "")
    .trim()
  return short.length > 80 ? `${short.slice(0, 80)}…` : short || "без описания"
}
