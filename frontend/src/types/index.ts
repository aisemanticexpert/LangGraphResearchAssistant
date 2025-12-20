export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

export interface QueryResponse {
  thread_id: string;
  final_response: string | null;
  interrupted: boolean;
  interrupt_info: InterruptInfo | null;
  error: string | null;
  cached: boolean;
}

export interface InterruptInfo {
  interrupted: boolean;
  question: string;
  original_query: string;
  type: string;
}

export interface ConversationState {
  thread_id: string;
  state: {
    detected_company: string | null;
    clarity_status: string;
    research_attempts: number;
    confidence_score: number;
    validation_result: string;
    final_response: string | null;
  };
}

export interface CacheStats {
  total_entries: number;
  valid_entries: number;
  max_size: number;
  ttl_seconds: number;
  enabled: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  cache_stats: CacheStats;
}

export interface CompanyListResponse {
  companies: string[];
  total: number;
}

export interface Conversation {
  threadId: string;
  messages: Message[];
  company: string | null;
  createdAt: Date;
}
