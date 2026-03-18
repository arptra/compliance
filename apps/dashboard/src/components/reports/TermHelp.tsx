import { glossary } from '../../lib/glossary'

export function TermHelp({ term }: { term: keyof typeof glossary | string }) {
  const text = glossary[term] ?? 'Пояснение недоступно.'
  return (
    <span className='term-help' title={text} aria-label={text}>
      ⓘ
    </span>
  )
}
