'use client';

import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  Plus, Send, Loader2, Trash2, MessageSquare,
  ChevronDown, AlertCircle, BookOpen, Download
} from 'lucide-react';
import {
  createSession, listSessions, getSession,
  sendMessage, deleteSession, exportSession,
  SessionResponse, SessionDetailResponse,
  MessageResponse, listDocuments, DocumentInfo
} from '@/lib/api';
import ChatMessage from '@/components/ChatMessage';
import clsx from 'clsx';

function ResearchContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [activeSession, setActiveSession] = useState<SessionDetailResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [showNewSession, setShowNewSession] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [initialized, setInitialized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const activeSessionIdRef = useRef<string | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const fetchSessions = useCallback(async () => {
    try {
      const data = await listSessions();
      setSessions(data);
      return data;
    } catch {
      return [];
    }
  }, []);

  const loadSession = useCallback(async (id: string, force = false) => {
    if (activeSessionIdRef.current === id && !force) return;
    activeSessionIdRef.current = id;
    setSessionLoading(true);
    setError(null);
    try {
      const data = await getSession(id);
      setActiveSession(data);
      if (data.doc_id) setSelectedDocId(data.doc_id);
      setTimeout(scrollToBottom, 100);
    } catch {
      setError('Failed to load session');
      activeSessionIdRef.current = null;
    } finally {
      setSessionLoading(false);
    }
  }, [scrollToBottom]);

  // Initialize once on mount
  useEffect(() => {
    if (initialized) return;
    setInitialized(true);

    const init = async () => {
      const [sessionsData, docsData] = await Promise.all([
        fetchSessions(),
        listDocuments().catch(() => ({ documents: [] })),
      ]);
      setDocuments((docsData as any).documents || []);

      const sid = searchParams.get('session');
      if (sid) {
        loadSession(sid);
      }
    };
    init();
  }, [initialized, fetchSessions, loadSession, searchParams]);

  // Scroll when messages change
  useEffect(() => {
    if (activeSession?.messages?.length) {
      setTimeout(scrollToBottom, 50);
    }
  }, [activeSession?.messages?.length, scrollToBottom]);

  // Handle URL session param changes (when clicking sidebar items)
  useEffect(() => {
    const sid = searchParams.get('session');
    if (sid && sid !== activeSessionIdRef.current) {
      loadSession(sid);
    }
  }, [searchParams, loadSession]);

  const handleSelectSession = useCallback((id: string) => {
    router.push(`/research?session=${id}`, { scroll: false });
    loadSession(id);
  }, [router, loadSession]);

  const handleNewSession = async () => {
    try {
      const doc = documents.find(d => d.doc_id === selectedDocId);
      const session = await createSession({
        title: newTitle.trim() || 'New research session',
        doc_id: selectedDocId || null,
        doc_name: doc?.file_name || null,
      });
      setShowNewSession(false);
      setNewTitle('');
      await fetchSessions();
      router.push(`/research?session=${session.id}`, { scroll: false });
      await loadSession(session.id, true);
    } catch {
      setError('Failed to create session');
    }
  };

  const handleSend = async () => {
    if (!input.trim() || sending || !activeSession) return;
    const content = input.trim();
    setInput('');
    setSending(true);
    setError(null);

    const tempId = `temp-${Date.now()}`;
    const tempUserMsg: MessageResponse = {
      id: tempId,
      session_id: activeSession.id,
      role: 'user',
      content,
      citations: [],
      created_at: new Date().toISOString(),
    };

    setActiveSession(prev => prev ? {
      ...prev,
      messages: [...prev.messages, tempUserMsg]
    } : prev);
    setTimeout(scrollToBottom, 50);

    try {
      const reply = await sendMessage(activeSession.id, content);
      setActiveSession(prev => prev ? {
        ...prev,
        messages: [
          ...prev.messages.filter(m => m.id !== tempId),
          { ...tempUserMsg, id: `user-${Date.now()}` },
          reply,
        ],
        message_count: (prev.message_count || 0) + 2,
      } : prev);
      await fetchSessions();
      setTimeout(scrollToBottom, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      setActiveSession(prev => prev ? {
        ...prev,
        messages: prev.messages.filter(m => m.id !== tempId)
      } : prev);
    } finally {
      setSending(false);
    }
  };

  const handleExport = async () => {
    if (!activeSession) return;
    try {
      await exportSession(activeSession.id, activeSession.title);
    } catch {
      setError('Export failed');
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm('Delete this session and all its messages?')) return;
    try {
      await deleteSession(id);
      if (activeSession?.id === id) {
        setActiveSession(null);
        activeSessionIdRef.current = null;
        router.push('/research', { scroll: false });
      }
      await fetchSessions();
    } catch {
      setError('Failed to delete session');
    }
  };

  return (
    <div className="flex h-screen overflow-hidden"
      style={{ marginLeft: '-2rem', marginTop: '-2rem', marginBottom: '-2rem' }}>

      {/* Session sidebar */}
      <div className="w-64 bg-white border-r border-slate-200 flex flex-col flex-shrink-0 h-full">
        <div className="p-4 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-800 mb-3">Research sessions</h2>
          <button
            onClick={() => setShowNewSession(v => !v)}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            New session
          </button>
        </div>

        {showNewSession && (
          <div className="p-3 border-b border-slate-100 bg-blue-50">
            <p className="text-xs font-medium text-slate-600 mb-2">Session details</p>
            <input
              value={newTitle}
              onChange={e => setNewTitle(e.target.value)}
              placeholder="Session title (optional)"
              className="w-full px-2.5 py-1.5 text-xs border border-slate-200 rounded-md mb-2 bg-white focus:outline-none focus:border-blue-400"
              onKeyDown={e => { if (e.key === 'Enter') handleNewSession(); }}
            />
            <div className="relative mb-2">
              <select
                value={selectedDocId}
                onChange={e => setSelectedDocId(e.target.value)}
                className="w-full appearance-none px-2.5 py-1.5 text-xs border border-slate-200 rounded-md bg-white pr-6 focus:outline-none focus:border-blue-400"
              >
                <option value="">All documents</option>
                {documents.map(d => (
                  <option key={d.doc_id} value={d.doc_id}>{d.file_name}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-2 top-1.5 w-3 h-3 text-slate-400 pointer-events-none" />
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleNewSession}
                className="flex-1 py-1.5 bg-blue-600 text-white rounded-md text-xs font-medium hover:bg-blue-700"
              >
                Create
              </button>
              <button
                onClick={() => setShowNewSession(false)}
                className="flex-1 py-1.5 bg-slate-100 text-slate-600 rounded-md text-xs hover:bg-slate-200"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto py-2">
          {sessions.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <MessageSquare className="w-6 h-6 text-slate-300 mx-auto mb-2" />
              <p className="text-xs text-slate-400">No sessions yet</p>
              <p className="text-[10px] text-slate-300 mt-1">Create one to get started</p>
            </div>
          ) : sessions.map(s => (
            <div
              key={s.id}
              onClick={() => handleSelectSession(s.id)}
              className={clsx(
                'group flex items-start gap-2 px-3 py-2.5 cursor-pointer transition-colors',
                activeSession?.id === s.id
                  ? 'bg-blue-50 border-r-2 border-blue-500'
                  : 'hover:bg-slate-50'
              )}
            >
              <div className="flex-1 min-w-0">
                <p className={clsx(
                  'text-xs font-medium truncate',
                  activeSession?.id === s.id ? 'text-blue-700' : 'text-slate-700'
                )}>
                  {s.title}
                </p>
                <p className="text-[10px] text-slate-400 mt-0.5 truncate">
                  {s.doc_name || 'All documents'} · {s.message_count} msg{s.message_count !== 1 ? 's' : ''}
                </p>
              </div>
              <button
                onClick={e => handleDeleteSession(e, s.id)}
                className="opacity-0 group-hover:opacity-100 p-0.5 text-slate-400 hover:text-red-500 transition-all mt-0.5 shrink-0"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-50">
        {!activeSession ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
            <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center mb-4">
              <BookOpen className="w-7 h-7 text-blue-400" />
            </div>
            <h2 className="text-base font-semibold text-slate-700 mb-1">Start researching</h2>
            <p className="text-sm text-slate-400 max-w-xs">
              Create a session to ask questions about your legal documents with full conversation history.
            </p>
            <button
              onClick={() => setShowNewSession(true)}
              className="mt-5 flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
            >
              <Plus className="w-4 h-4" />
              New session
            </button>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="bg-white border-b border-slate-200 px-5 py-3 flex items-center justify-between flex-shrink-0">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">{activeSession.title}</h2>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {activeSession.doc_name || 'All documents'} · {activeSession.message_count} messages
                </p>
              </div>
              <div className="flex items-center gap-2">
                {activeSession.doc_name && (
                  <span className="px-2.5 py-1 bg-blue-50 border border-blue-100 rounded-full text-xs font-medium text-blue-700 truncate max-w-[200px]">
                    {activeSession.doc_name}
                  </span>
                )}
                <button
                  onClick={handleExport}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 bg-slate-100 border border-slate-200 rounded-lg hover:bg-slate-200 transition-colors"
                  title="Export session as markdown"
                >
                  <Download className="w-3 h-3" />
                  Export
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
              {sessionLoading ? (
                <div className="flex justify-center py-10">
                  <Loader2 className="w-5 h-5 text-slate-400 animate-spin" />
                </div>
              ) : activeSession.messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center py-16">
                  <MessageSquare className="w-8 h-8 text-slate-300 mb-3" />
                  <p className="text-sm text-slate-400">Ask your first question below</p>
                  <p className="text-xs text-slate-300 mt-1">The conversation history is saved automatically</p>
                </div>
              ) : (
                activeSession.messages.map(msg => (
                  <ChatMessage key={msg.id} message={msg} />
                ))
              )}

              {sending && (
                <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 w-fit">
                  <div className="flex gap-1">
                    {[0, 1, 2].map(i => (
                      <div
                        key={i}
                        className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                  </div>
                  <span className="text-xs text-slate-400">LexMind is researching...</span>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {error && (
              <div className="mx-5 mb-2 flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-3.5 h-3.5 text-red-500 mt-0.5 shrink-0" />
                <p className="text-xs text-red-700">{error}</p>
              </div>
            )}

            {/* Input */}
            <div className="bg-white border-t border-slate-200 px-4 py-3 flex-shrink-0">
              <div className="flex items-end gap-3">
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  placeholder="Ask a follow-up question... (Enter to send, Shift+Enter for new line)"
                  rows={1}
                  className="flex-1 px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 resize-none transition-colors"
                  style={{ maxHeight: '120px' }}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  onInput={e => {
                    const t = e.target as HTMLTextAreaElement;
                    t.style.height = 'auto';
                    t.style.height = Math.min(t.scrollHeight, 120) + 'px';
                  }}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || sending}
                  className={clsx(
                    'p-2.5 rounded-xl transition-all flex-shrink-0',
                    !input.trim() || sending
                      ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95'
                  )}
                >
                  {sending
                    ? <Loader2 className="w-4 h-4 animate-spin" />
                    : <Send className="w-4 h-4" />
                  }
                </button>
              </div>
              <p className="text-[10px] text-slate-400 mt-1.5">
                Context from this session is included automatically
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function ResearchPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    }>
      <ResearchContent />
    </Suspense>
  );
}
