import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'

type TaxonomyLabelsResponse = {
  category_labels: Record<string, string>
  subcategory_labels: Record<string, Record<string, string>>
}

export function useTaxonomyLabels() {
  const q = useQuery({
    queryKey: ['taxonomy-labels'],
    queryFn: () => apiGet<TaxonomyLabelsResponse>('/api/meta/taxonomy-labels'),
    staleTime: 5 * 60_000,
  })

  const categoryLabel = (category?: string | null) => {
    if (!category) return ''
    return q.data?.category_labels?.[category] ?? category
  }

  const subcategoryLabel = (category?: string | null, subcategory?: string | null) => {
    if (!subcategory) return ''
    if (!category) return subcategory
    return q.data?.subcategory_labels?.[category]?.[subcategory] ?? subcategory
  }

  return { ...q, categoryLabel, subcategoryLabel }
}
