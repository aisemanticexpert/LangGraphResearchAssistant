import axios from 'axios';
import {
  QueryResponse,
  ConversationState,
  HealthResponse,
  CompanyListResponse,
  CacheStats,
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const researchApi = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },

  // Get list of available companies
  async getCompanies(): Promise<CompanyListResponse> {
    const response = await api.get<CompanyListResponse>('/companies');
    return response.data;
  },

  // Process a new query
  async query(query: string, useCache: boolean = true): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/query', {
      query,
      use_cache: useCache,
    });
    return response.data;
  },

  // Continue an existing conversation
  async continueConversation(threadId: string, query: string): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/continue', {
      thread_id: threadId,
      query,
    });
    return response.data;
  },

  // Provide clarification for interrupted query
  async clarify(threadId: string, clarification: string): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/clarify', {
      thread_id: threadId,
      clarification,
    });
    return response.data;
  },

  // Get conversation state
  async getConversationState(threadId: string): Promise<ConversationState> {
    const response = await api.get<ConversationState>(`/conversation/${threadId}`);
    return response.data;
  },

  // Export conversation
  async exportConversation(threadId: string, format: 'json' | 'markdown' = 'json'): Promise<{ success: boolean; filepath: string; format: string }> {
    const response = await api.post('/export', {
      thread_id: threadId,
      format,
    });
    return response.data;
  },

  // Get cache stats
  async getCacheStats(): Promise<CacheStats> {
    const response = await api.get<CacheStats>('/cache/stats');
    return response.data;
  },

  // Clear cache
  async clearCache(): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/cache/clear');
    return response.data;
  },
};

export default researchApi;
