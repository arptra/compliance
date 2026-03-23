export function ReviewCommentDialog({ value, onChange }: { value?: string; onChange: (v:string)=>void }) {
  return <input placeholder='comment' value={value ?? ''} onChange={(e) => onChange(e.target.value)} style={{ width: 160 }} />
}
