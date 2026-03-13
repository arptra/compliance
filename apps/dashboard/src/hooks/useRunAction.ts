import { useMutation } from '@tanstack/react-query'
import { apiPost } from '../lib/api'

export function useRunAction(path: string) {
  return useMutation({ mutationFn: (params: Record<string, unknown>) => apiPost(path, { params }) })
}
