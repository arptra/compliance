export function ModelVersionBadge({ version }: { version?: string | null }) {
  return <span style={{ padding: '2px 8px', borderRadius: 10, background: '#e2e8f0' }}>{version ?? 'no-model'}</span>
}
