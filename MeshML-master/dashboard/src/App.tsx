import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@/components/ThemeProvider';
import { ToastProvider } from '@/components/Toast';
import ErrorBoundary from '@/components/ErrorBoundary';
import DashboardLayout from '@/layouts/DashboardLayout';

import Login from '@/views/Login';
import Workspace from '@/views/Workspace';
import GroupDashboard from '@/views/GroupDashboard';
import Cockpit from '@/views/Cockpit';


// Create a client for react-query
const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="meshml-theme">
        <ToastProvider>
          <ErrorBoundary>
            <BrowserRouter>
              <Routes>
                {/* Login Page */}
                <Route path="/login" element={<Login />} />
                
                {/* Dashboard Layout wrapper for internal views */}
                <Route element={<DashboardLayout />}>
                  <Route path="/workspace" element={<Workspace />} />
                  <Route path="/groups/:groupId" element={<GroupDashboard />} />

                  <Route path="/jobs/:id/live" element={<Cockpit />} />
                </Route>

                {/* Redirect root to login for now */}
                <Route path="/" element={<Navigate to="/login" replace />} />
              </Routes>
            </BrowserRouter>
          </ErrorBoundary>
        </ToastProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
