'use client';

import { useState, useEffect } from 'react';
import {
  Search, Loader2, AlertCircle, ChevronDown,
  BookOpen, CheckCircle, XCircle, RotateCcw
} from 'lucide-react';
import { queryDocument, listDocuments, DocumentInfo, QueryResponse } from '@/lib/api';
import clsx from 'clsx';

const QUERY_TYPE_STYLES: Record<string, string> = {
  factual:       'bg-blue-50 text-blue-700 border-blue-100',
  analytical:    'bg-purple-50 text-purple-700 border-purple-100',
  comparison:    'bg-amber-50 text-amber-700 border-amber-100',
  summarisation: 'bg-green-50 text-green-700 border-green-100',
};

function ScoreBadge({ label, score }: { label: string; score: number | null }) {
  if (score === null) return null;
  const pct = Math.round(score * 100);
  const color = pct >= 80
    ? 'bg-green-50 text-green-700 border-green-100'
    : pct >= 60
    ? 'bg-amber-50 text-amber-700 border-amber-100'
    : 'bg-red-50 text-red-700 border-red-100';
  return (
    <span className={clsx('inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-xs font-medium', color)}>
      <span className="opacity-60">{label}</span>
      <span className="font-semibold">{pct}%</span>
    </span>
  );
}

function highlightCitations(text: string, citations: string[]) {
  if (!citations.length) return <>{text}</>;
  const escaped = citations.map(c => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const regex = new RegExp(`\\[(${escaped.join('|')})\\]`, 'gi');
  const parts = text.split(regex);
  return (
    <>
      {parts.map((part, i) =>
        citations.some(c => c.toLowerCase() === part.toLowerCase()) ? (
          <span key={i} className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-800 mx-0.5">
            {part}
          </span>
        ) : part
      )}
    </>
  );
}

export default function QueryPage() {
  const [query, setQuery] = useState('');
  const [selectedDocId, setSelectedDocId] = useState('');
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [docsLoading, setDocsLoading] = useState(true);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then(data => setDocuments(data.documents))
      .catch(() => {})
      .finally(() => setDocsLoading(false));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await queryDocument({
        query: query.trim(),
        doc_id: selectedDocId || null,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-slate-900">Query</h1>
        <p className="text-sm text-slate-500 mt-1">Ask a question about your legal documents</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Document selector */}
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1.5">Document</label>
          <div className="relative">
            <select
              value={selectedDocId}
              onChange={e => setSelectedDocId(e.target.value)}
              className="w-full appearance-none bg-white border border-slate-200 rounded-lg px-3.5 py-2.5 pr-9 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors"
            >
              <option value="">All documents</option>
              {docsLoading ? (
                <option disabled>Loading...</option>
              ) : (
                documents.map(doc => (
                  <option key={doc.doc_id} value={doc.doc_id}>
                    {doc.file_name} ({doc.chunk_count} chunks)
                  </option>
                ))
              )}
            </select>
            <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-slate-400 pointer-events-none" />
          </div>
        </div>

        {/* Question input */}
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1.5">Your question</label>
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="e.g. What are the termination conditions?"
            rows={3}
            className="w-full bg-white border border-slate-200 rounded-lg px-3.5 py-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors resize-none"
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                handleSubmit(e as any);
              }
            }}
          />
          <p className="text-[11px] text-slate-400 mt-1">Ctrl+Enter to submit</p>
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className={clsx(
            'flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all',
            !query.trim() || loading
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-[0.98]'
          )}
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" />Researching...</>
          ) : (
            <><Search className="w-4 h-4" />Ask LexMind</>
          )}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="mt-6 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-700">Query failed</p>
            <p className="text-sm text-red-600 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-8 space-y-3">
          {/* Meta row */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <BookOpen className="w-3.5 h-3.5" />
              {result.chunks_used} chunks used
              {result.regeneration_count > 0 && (
                <span className="ml-1 flex items-center gap-1">
                  <RotateCcw className="w-3 h-3" />
                  {result.regeneration_count} regen
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className={clsx(
                'px-2.5 py-0.5 rounded-full text-xs font-medium border capitalize',
                QUERY_TYPE_STYLES[result.query_type] || 'bg-slate-50 text-slate-600 border-slate-200'
              )}>
                {result.query_type}
              </span>
              {result.critique_passed !== null && (
                result.critique_passed
                  ? <CheckCircle className="w-4 h-4 text-green-500" />
                  : <XCircle className="w-4 h-4 text-red-500" />
              )}
            </div>
          </div>

          {/* Answer */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <p className="text-sm leading-7 text-slate-800 whitespace-pre-wrap">
              {highlightCitations(result.final_answer, result.citations)}
            </p>
          </div>

          {/* Citations */}
          {result.citations.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-medium text-slate-400">Cited:</span>
              {result.citations.map(c => (
                <span key={c} className="px-2.5 py-1 bg-blue-50 border border-blue-100 rounded-full text-xs font-medium text-blue-700">
                  {c}
                </span>
              ))}
            </div>
          )}

          {/* Quality scores */}
          {(result.groundedness_score !== null || result.citation_score !== null || result.relevance_score !== null) && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-medium text-slate-400">Quality:</span>
              <ScoreBadge label="Grounded" score={result.groundedness_score} />
              <ScoreBadge label="Citations" score={result.citation_score} />
              <ScoreBadge label="Relevant" score={result.relevance_score} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
