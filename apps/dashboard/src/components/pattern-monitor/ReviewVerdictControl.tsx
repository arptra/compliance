export function ReviewVerdictControl({ value, onChange }: { value?: string; onChange: (v:'true'|'false'|'uncertain')=>void }) {
  const btn = (v:'true'|'false'|'uncertain', label: string) => <button style={{ fontWeight: value===v ? 700:400 }} onClick={() => onChange(v)}>{label}</button>
  return <div style={{ display: 'flex', gap: 4 }}>{btn('true', '✅ True')}{btn('false', '❌ False')}{btn('uncertain', '❓ Uncertain')}</div>
}
