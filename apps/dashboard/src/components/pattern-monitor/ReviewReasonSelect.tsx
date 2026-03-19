const reasons = ['normal_background','duplicate','wrong_cluster','weak_similarity','noisy_text','valid_special_pattern','other']

export function ReviewReasonSelect({ value, onChange }: { value?: string; onChange: (v: string)=>void }) {
  return <select value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
    <option value=''>reason</option>
    {reasons.map((r) => <option key={r} value={r}>{r}</option>)}
  </select>
}
