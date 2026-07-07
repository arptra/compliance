import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { cancelGigaChatBackgroundTask, listGigaChatBackgroundTasks } from '../features/gigachat/api'
import type { GigaChatBackgroundTaskSummary } from '../features/gigachat/types'

function formatDate(value?: string | null) {
  return value ? new Date(value).toLocaleString() : '—'
}

function statusLabel(status: GigaChatBackgroundTaskSummary['status']) {
  const labels: Record<GigaChatBackgroundTaskSummary['status'], string> = {
    queued: 'В очереди',
    running: 'В работе',
    completed: 'Готово',
    cancelled: 'Отменено',
    failed: 'Ошибка',
  }
  return labels[status]
}

export default function GigaChatBackgroundTasksPage() {
  const qc = useQueryClient()
  const tasksQ = useQuery({
    queryKey: ['gigachat-background-tasks'],
    queryFn: listGigaChatBackgroundTasks,
    refetchInterval: 2000,
  })
  const cancelTask = useMutation({
    mutationFn: cancelGigaChatBackgroundTask,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['gigachat-background-tasks'] }),
  })

  return <div className='transport-page'>
    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h2>Фоновые задачи GigaChat</h2>
          <p>Здесь видны запущенные фоновые разметки: статус, прогресс, время старта и быстрые действия.</p>
        </div>
        <Link className='transport-link-button' to='/gigachat'>Вернуться в Lab</Link>
      </div>
    </section>

    <section className='card transport-result'>
      {tasksQ.isLoading ? <div>Загружаем фоновые задачи...</div> : null}
      {tasksQ.isError ? <div className='transport-error'>{(tasksQ.error as Error).message}</div> : null}
      {!tasksQ.isLoading && !tasksQ.data?.tasks.length ? <div className='lab-muted'>Фоновых задач пока нет.</div> : null}

      {tasksQ.data?.tasks.length ? <div className='background-task-list'>
        {tasksQ.data.tasks.map((task) => {
          const running = task.status === 'queued' || task.status === 'running'
          const progressPercent = Math.round((task.progress || 0) * 100)
          return <article key={task.task_id} className='background-task-card'>
            <div className='background-task-head'>
              <div>
                <h3>{task.filename || task.task_id}</h3>
                <div className='lab-muted'>Лист: <code>{task.sheet_name || '—'}</code> · ID: <code>{task.task_id}</code></div>
              </div>
              <span className={`background-task-status ${task.status}`}>{statusLabel(task.status)}</span>
            </div>

            <div className='background-task-progress'>
              <div className='giga-processing-progressbar'>
                <div className='giga-processing-progressbar-fill' style={{ width: `${progressPercent}%` }} />
              </div>
              <div className='lab-muted'>{task.completed_rows} из {task.total_rows} строк · {progressPercent}%</div>
            </div>

            <div className='background-task-meta'>
              <div><span>Создана</span><strong>{formatDate(task.created_at)}</strong></div>
              <div><span>Старт</span><strong>{formatDate(task.started_at)}</strong></div>
              <div><span>Финиш</span><strong>{formatDate(task.finished_at)}</strong></div>
              <div><span>Ошибок строк</span><strong>{task.failed_rows}</strong></div>
              <div><span>Workers</span><strong>{task.async_workers || 1}</strong></div>
            </div>

            <div className='lab-muted'>{task.current_label || '—'}</div>
            {task.error ? <pre className='background-task-error'>{task.error}</pre> : null}

            <div className='transport-actions'>
              {running ? <button type='button' onClick={() => cancelTask.mutate(task.task_id)} disabled={cancelTask.isPending}>
                Отменить
              </button> : null}
              {task.status === 'completed' ? <Link className='transport-link-button' to={`/gigachat?backgroundTaskId=${encodeURIComponent(task.task_id)}`}>
                Перейти в рабочую тетрадь
              </Link> : null}
            </div>
          </article>
        })}
      </div> : null}
    </section>
  </div>
}
