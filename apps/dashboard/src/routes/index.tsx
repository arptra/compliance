import { createBrowserRouter } from 'react-router-dom'
import { Layout } from '../components/Layout'
import OverviewPage from '../pages/OverviewPage'
import CategoriesPage from '../pages/CategoriesPage'
import TimeseriesPage from '../pages/TimeseriesPage'
import PreparationPage from '../pages/PreparationPage'
import PatternFitPage from '../pages/PatternFitPage'
import PatternMonitorPage from '../pages/PatternMonitorPage'
import ReportsPage from '../pages/ReportsPage'
import SettingsPage from '../pages/SettingsPage'

export const router = createBrowserRouter([
  { path: '/', element: <Layout />, children: [
    { index: true, element: <OverviewPage /> },
    { path: 'overview', element: <OverviewPage /> },
    { path: 'categories', element: <CategoriesPage /> },
    { path: 'timeseries', element: <TimeseriesPage /> },
    { path: 'preparation', element: <PreparationPage /> },
    { path: 'pattern-fit', element: <PatternFitPage /> },
    { path: 'pattern-monitor', element: <PatternMonitorPage /> },
    { path: 'reports', element: <ReportsPage /> },
    { path: 'settings', element: <SettingsPage /> }
  ] }
])
