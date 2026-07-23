const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface IngestResponse {
  success: boolean;
  doc_id: string;
  file_name: string;
  doc_title: string;
  chunks_created: number;
  articles_found: number;
  total_chars: number;
  message: string;
}

export interface QueryRequest {
  query: string;
  doc_id?: string | null;
}

export interface QueryResponse {
  success: boolean;
  query: string;
  query_type: string;
  final_answer: string;
  citations: string[];
  groundedness_score: number | null;
  citation_score: number | null;
  relevance_score: number | null;
  critique_passed: boolean | null;
  regeneration_count: number;
  chunks_used: number;
  error: string | null;
}

export interface DocumentInfo {
  doc_id: string;
  file_name: string;
  chunk_count: number;
}

export interface DocumentListResponse {
  total_documents: number;
  total_chunks: number;
  documents: DocumentInfo[];
}

export interface HealthResponse {
  status: string;
  version: string;
  documents_ingested: number;
  total_chunks: number;
}

export async function ingestDocument(
  file: File,
  docTitle?: string
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (docTitle && docTitle.trim()) {
    formData.append('doc_title', docTitle.trim());
  }
  const response = await fetch(`${API_BASE}/api/ingest`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || 'Upload failed');
  }
  return response.json();
}

export async function queryDocument(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Query failed' }));
    throw new Error(error.detail || 'Query failed');
  }
  return response.json();
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const response = await fetch(`${API_BASE}/api/documents`);
  if (!response.ok) throw new Error('Failed to fetch documents');
  return response.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete document');
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('API unavailable');
  return response.json();
}


// ─── Session Types ────────────────────────────────────────────

export interface MessageResponse {
  id: string;
  session_id: string;
  role: string;
  content: string;
  query_type?: string;
  citations: string[];
  groundedness_score?: number | null;
  citation_score?: number | null;
  relevance_score?: number | null;
  critique_passed?: boolean | null;
  regeneration_count?: number;
  chunks_used?: number;
  created_at: string;
}

export interface SessionResponse {
  id: string;
  title: string;
  doc_id?: string | null;
  doc_name?: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionDetailResponse extends SessionResponse {
  messages: MessageResponse[];
}


// ─── Session Functions ────────────────────────────────────────

export async function createSession(data: {
  title?: string;
  doc_id?: string | null;
  doc_name?: string | null;
}): Promise<SessionResponse> {
  const response = await fetch(`${API_BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'New research session', ...data }),
  });
  if (!response.ok) throw new Error('Failed to create session');
  return response.json();
}

export async function listSessions(): Promise<SessionResponse[]> {
  const response = await fetch(`${API_BASE}/api/sessions`);
  if (!response.ok) throw new Error('Failed to fetch sessions');
  return response.json();
}

export async function getSession(id: string): Promise<SessionDetailResponse> {
  const response = await fetch(`${API_BASE}/api/sessions/${id}`);
  if (!response.ok) throw new Error('Failed to fetch session');
  return response.json();
}

export async function sendMessage(
  sessionId: string,
  content: string
): Promise<MessageResponse> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(error.detail || 'Failed to send message');
  }
  return response.json();
}

export async function deleteSession(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/sessions/${id}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete session');
}

export interface RecentQuery {
  id: string;
  session_id: string;
  content: string;
  created_at: string;
}

export async function getRecentQueries(limit = 10): Promise<RecentQuery[]> {
  const response = await fetch(
    `${API_BASE}/api/sessions/recent-queries?limit=${limit}`
  );
  if (!response.ok) return [];
  return response.json();
}

export async function exportSession(
  sessionId: string,
  title: string
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/export`
  );
  if (!response.ok) throw new Error('Export failed');

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = title.replace(/\s+/g, '_').slice(0, 50) + '.md';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function reingestDocument(
  docId: string,
  file: File,
  docTitle?: string
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (docTitle?.trim()) {
    formData.append('doc_title', docTitle.trim());
  }
  const response = await fetch(
    `${API_BASE}/api/documents/${docId}/reingest`,
    { method: 'PUT', body: formData }
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(error.detail || 'Re-ingest failed');
  }
  return response.json();
}
