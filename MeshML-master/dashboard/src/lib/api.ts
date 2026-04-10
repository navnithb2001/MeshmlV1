import axios from 'axios';

// ==========================================
// 1. Interfaces Based on Python Pydantic Schemas
// ==========================================

export interface UserResponse {
  id: string; // uuid
  email: string;
  full_name: string | null;
  is_active: boolean;
  created_at: string; // datetime
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: UserResponse;
}

export interface DatasetUploadResponse {
  dataset_id: string;
  name: string;
  format: string;
  status: string;
  total_size_bytes: number | null;
  file_count: number | null;
  num_samples: number | null;
  num_classes: number | null;
  local_path: string | null;
  gcs_path: string | null;
  message: string;
  uploaded_at: string;
}

export interface DatasetResponse {
  id: string;
  name: string;
  format: string;
  upload_type: string;
  source_url: string | null;
  local_path: string | null;
  gcs_path: string | null;
  total_size_bytes: number | null;
  file_count: number | null;
  num_samples: number | null;
  num_classes: number | null;
  num_shards: number | null;
  shard_strategy: string | null;
  sharded_at: string | null;
  status: string;
  error_message: string | null;
  metadata: Record<string, any> | null;
  uploaded_by: string;
  created_at: string;
  updated_at: string;
}

export interface DatasetListResponse {
  datasets: DatasetResponse[];
  total: number;
}

export interface JobResponse {
  id: string;
  group_id: string;
  model_id: string | null;
  dataset_id: string | null;
  config: Record<string, any> | null;
  status: string;
  progress: Record<string, any> | null;
  error_message: string | null;
  created_by: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobProgressResponse {
  job_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  current_batch: number;
  total_batches: number;
  loss: number;
  accuracy: number;
  worker_count: number;
}

export interface WorkerResponse {
  id: string;
  worker_id: string;
  user_email: string | null;
  capabilities: Record<string, any> | null;
  status: string;
  last_heartbeat: string | null;
  created_at: string;
}

export interface GroupResponse {
  id: string;
  name: string;
  description: string | null;
  is_public: boolean;
  owner_id: string;
  created_at: string;
}

export interface GroupMemberResponse {
  id: string;
  group_id: string;
  user_id: string | null;
  worker_id: string | null;
  role: string;
  joined_at: string;
  user?: UserResponse | null;
}

export interface InvitationResponse {
  code: string;
  group_id: string;
  max_uses: number | null;
  current_uses: number;
  expires_at: string;
  created_at: string;
}

// ==========================================
// 2. Axios Auto-Inject Setup
// ==========================================

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token && config.headers) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

import { toastEmitter } from '@/components/Toast';

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    } else if (error.response?.status === 500) {
      toastEmitter.emit('Server error. Please try again later.', 'error');
    } else if (!error.response) {
      toastEmitter.emit('Network error. Check your connection.', 'error');
    }
    return Promise.reject(error);
  }
);

export default api;

// ==========================================
// 3. API Fetching Functions
// ==========================================

export const groupsAPI = {
  createGroup: async (data: { name: string; description?: string }): Promise<GroupResponse> => {
    const res = await api.post('/groups', data);
    return res.data;
  },

  listGroups: async (): Promise<GroupResponse[]> => {
    const res = await api.get('/groups');
    return res.data.groups;
  },

  getGroup: async (id: string): Promise<GroupResponse> => {
    const res = await api.get(`/groups/${id}`);
    return res.data;
  },

  createInvitation: async (groupId: string, data: { max_uses?: number; expires_in_hours?: number }): Promise<InvitationResponse> => {
    const res = await api.post(`/invitations/${groupId}/invitations`, data);
    return res.data;
  },

  acceptInvitation: async (data: { worker_id: string; invitation_code: string }): Promise<void> => {
    await api.post('/invitations/accept', data);
  },

  updateGroup: async (groupId: string, data: { name?: string; description?: string; is_public?: boolean }): Promise<GroupResponse> => {
    const res = await api.put(`/groups/${groupId}`, data);
    return res.data;
  },

  getGroupMembers: async (groupId: string): Promise<GroupMemberResponse[]> => {
    const res = await api.get(`/groups/${groupId}/members`);
    return res.data;
  },

  updateMemberRole: async (groupId: string, userId: string, role: string): Promise<GroupMemberResponse> => {
    const res = await api.put(`/groups/${groupId}/members/${userId}/role`, { role });
    return res.data;
  },

  removeMember: async (groupId: string, userId: string): Promise<void> => {
    await api.delete(`/groups/${groupId}/members/${userId}`);
  },

  deleteGroup: async (groupId: string): Promise<void> => {
    await api.delete(`/groups/${groupId}`);
  }
};

export const workersAPI = {
  listWorkers: async (params?: { group_id?: string }): Promise<WorkerResponse[]> => {
    const res = await api.get('/workers', { params });
    return res.data;
  }
};

export const authAPI = {
  login: async (credentials: any): Promise<TokenResponse> => {
    const res = await api.post('/auth/login', credentials);
    return res.data;
  },
  
  register: async (userData: any): Promise<UserResponse> => {
    const res = await api.post('/auth/register', userData);
    return res.data;
  },

  getCurrentUser: async (): Promise<UserResponse> => {
    const res = await api.get('/auth/me');
    return res.data;
  },

  refreshToken: async (): Promise<TokenResponse> => {
    const res = await api.post<TokenResponse>('/auth/refresh');
    return res.data;
  },

  changePassword: async (data: any): Promise<any> => {
    const res = await api.post('/auth/password', data);
    return res.data;
  }
};

export const datasetsAPI = {
  uploadDataset: async (
    files: File[],
    strategy: string = "stratified",
    datasetName?: string,
    datasetFormat?: 'imagefolder' | 'csv' | 'coco',
    onUploadProgress?: (progressEvent: any) => void
  ): Promise<DatasetUploadResponse> => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    formData.append('shard_strategy', strategy);
    if (datasetName) formData.append('dataset_name', datasetName);
    if (datasetFormat) formData.append('dataset_format', datasetFormat);

    const res = await api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress
    });
    return res.data;
  },

  listDatasets: async (params?: { format?: string; status?: string }): Promise<DatasetListResponse> => {
    const res = await api.get('/datasets', { params });
    return res.data;
  },

  getDataset: async (id: string): Promise<DatasetResponse> => {
    const res = await api.get(`/datasets/${id}`);
    return res.data;
  },

  deleteDataset: async (id: string): Promise<void> => {
    await api.delete(`/datasets/${id}`);
  }
};

export const jobsAPI = {
  createJob: async (jobData: { group_id: string; model_id?: string; dataset_id?: string; config?: any }): Promise<JobResponse> => {
    const res = await api.post('/jobs', jobData);
    return res.data;
  },

  listJobs: async (params?: { group_id?: string }): Promise<JobResponse[]> => {
    const res = await api.get('/jobs', { params });
    return res.data;
  },

  getJob: async (id: string): Promise<JobResponse> => {
    const res = await api.get(`/jobs/${id}`);
    return res.data;
  },

  cancelJob: async (id: string): Promise<void> => {
    await api.delete(`/jobs/${id}`);
  },

  getJobProgress: async (id: string): Promise<JobProgressResponse> => {
    const res = await api.get(`/jobs/${id}/progress`);
    return res.data;
  }
};

export const modelsAPI = {
  uploadModelArchitecture: async (
    file: File,
    name: string,
    groupId: string,
  ): Promise<void> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    formData.append('group_id', groupId);
    await api.post('/models/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  getDownloadSignedUrl: async (modelId: string): Promise<any> => {
    const res = await api.get(`/models/${modelId}/download`);
    return res.data;
  }
};
