import { createBrowserRouter } from 'react-router-dom'
import { Layout } from '../components/Layout'
import OverviewPage from '../pages/OverviewPage'
import CategoriesPage from '../pages/CategoriesPage'
import TimeseriesPage from '../pages/TimeseriesPage'
import PreparationPage from '../pages/PreparationPage'
import GigaChatPage from '../pages/GigaChatPage'
import GigaChatBackgroundTasksPage from '../pages/GigaChatBackgroundTasksPage'
import PatternFitPage from '../pages/PatternFitPage'
import PatternMonitorPage from '../pages/PatternMonitorPage'
import ReviewDatasetPage from '../pages/ReviewDatasetPage'
import ModelQualityPage from '../pages/ModelQualityPage'
import ParquetViewerPage from '../pages/ParquetViewerPage'
import ReportsPage from '../pages/ReportsPage'
import SettingsPage from '../pages/SettingsPage'

export const router = createBrowserRouter([
  { path: '/', element: <Layout />, children: [
    { index: true, element: <OverviewPage /> },
    { path: 'overview', element: <OverviewPage /> },
    { path: 'categories', element: <CategoriesPage /> },
    { path: 'timeseries', element: <TimeseriesPage /> },
    { path: 'preparation', element: <PreparationPage /> },
    { path: 'gigachat', element: <GigaChatPage /> },
    { path: 'gigachat/background', element: <GigaChatBackgroundTasksPage /> },
    { path: 'pattern-fit', element: <PatternFitPage /> },
    { path: 'pattern-monitor', element: <PatternMonitorPage /> },
    { path: 'review-dataset', element: <ReviewDatasetPage /> },
    { path: 'model-quality', element: <ModelQualityPage /> },
    { path: 'parquet-viewer', element: <ParquetViewerPage /> },
    { path: 'reports', element: <ReportsPage /> },
    { path: 'settings', element: <SettingsPage /> }
  ] }
])
