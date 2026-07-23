'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  FileText, Layers, Search, Upload,
  TrendingUp, AlertCircle, ArrowRight
} from 'lucide-react';
import { listDocuments, checkHealth, DocumentInfo,
         getRecentQueries, RecentQuery } from '@/lib/api';
import clsx from 'clsx';

function StatCard({
  label, value, sub, icon: Icon, color
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-500">{label}</p>
          <p className="text-2xl font-semibold text-slate-900 mt-1">{value}</p>
          {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
        </div>
        <div className={clsx('w-9 h-9 rounded-lg flex items-center justify-center', color)}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
    </div>
  );
}

function DocTypeBadge({ name }: { name: string }) {
  const ext = name.split('.').pop()?.toLowerCase();
  const styles: Record<string, string> = {
    pdf:  'bg-red-50 text-red-600 border-red-100',
    docx: 'bg-blue-50 text-blue-600 border-blue-100',
    txt:  'bg-slate-50 text-slate-600 border-slate-200',
  };
  return (
    <span className={clsx(
      'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border uppercase',
      styles[ext || ''] || styles.txt
    )}>
      {ext}
    </span>
  );
}

export default function DashboardPage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState<'ok' | 'degraded' | 'unknown'>('unknown');
  const [recentQueries, setRecentQueries] = useState<RecentQuery[]>([]);

  useEffect(() => {
    Promise.all([
      listDocuments(),
      checkHealth(),
      getRecentQueries(5),
    ]).then(([docsData, health, queries]) => {
      setDocs(docsData.documents);
      setTotalChunks(docsData.total_chunks);
      setApiStatus(health.status as 'ok' | 'degraded');
      setRecentQueries(queries);
    }).catch(() => {
      setApiStatus('degraded');
    }).finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-8 max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-slate-900">Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">Knowledge base overview</p>
      </div>

      {/* API Status Banner */}
      {apiStatus === 'degraded' && (
        <div className="mb-6 flex items-center gap-2 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg">
          <AlertCircle className="w-4 h-4 text-amber-500 shrink-0" />
          <p className="text-sm text-amber-700">
            Backend API is unavailable. Make sure the FastAPI server is running on port 8000.
          </p>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard
          label="Documents"
          value={loading ? '—' : docs.length}
          sub={docs.length === 1 ? '1 document' : `${docs.length} documents`}
          icon={FileText}
          color="bg-blue-50 text-blue-500"
        />
        <StatCard
          label="Total chunks"
          value={loading ? '—' : totalChunks}
          sub="Searchable units"
          icon={Layers}
          color="bg-purple-50 text-purple-500"
        />
        <StatCard
          label="API status"
          value={apiStatus === 'ok' ? 'Online' : apiStatus === 'degraded' ? 'Offline' : '—'}
          sub="FastAPI backend"
          icon={TrendingUp}
          color={apiStatus === 'ok' ? 'bg-green-50 text-green-500' : 'bg-red-50 text-red-500'}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <Link href="/upload" className="group bg-white border border-slate-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-sm transition-all">
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 bg-blue-50 rounded-lg flex items-center justify-center">
              <Upload className="w-4 h-4 text-blue-500" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-blue-400 transition-colors" />
          </div>
          <p className="font-medium text-slate-900 text-sm">Upload document</p>
          <p className="text-xs text-slate-400 mt-0.5">Add PDF, DOCX, or TXT to knowledge base</p>
        </Link>
        <Link href="/research" className="group bg-white border border-slate-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-sm transition-all">
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 bg-green-50 rounded-lg flex items-center justify-center">
              <Search className="w-4 h-4 text-green-500" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-green-400 transition-colors" />
          </div>
          <p className="font-medium text-slate-900 text-sm">Ask a question</p>
          <p className="text-xs text-slate-400 mt-0.5">Query your documents with AI agents</p>
        </Link>
      </div>

      {/* Recent Documents */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-slate-700">Recent documents</h2>
          <Link href="/documents" className="text-xs text-blue-500 hover:text-blue-700">
            View all →
          </Link>
        </div>

        {loading ? (
          <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
            <p className="text-sm text-slate-400">Loading...</p>
          </div>
        ) : docs.length === 0 ? (
          <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
            <FileText className="w-8 h-8 text-slate-300 mx-auto mb-3" />
            <p className="text-sm font-medium text-slate-600">No documents yet</p>
            <p className="text-xs text-slate-400 mt-1">Upload a legal document to get started</p>
            <Link href="/upload" className="inline-flex items-center gap-1 mt-3 text-xs text-blue-500 hover:text-blue-700">
              Upload now <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        ) : (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            {docs.slice(0, 5).map((doc, i) => (
              <div
                key={doc.doc_id}
                className={clsx(
                  'flex items-center gap-4 px-5 py-4',
                  i < docs.length - 1 && 'border-b border-slate-100'
                )}
              >
                <DocTypeBadge name={doc.file_name} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-900 truncate">
                    {doc.file_name}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5 font-mono truncate">
                    {doc.doc_id}
                  </p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-sm font-medium text-slate-700">{doc.chunk_count}</p>
                  <p className="text-xs text-slate-400">chunks</p>
                </div>
                <span className="shrink-0 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-100">
                  Ready
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Queries */}
      <div className="mt-8">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-slate-700">
            Recent queries
          </h2>
          <Link
            href="/research"
            className="text-xs text-blue-500 hover:text-blue-700"
          >
            New session →
          </Link>
        </div>

        {recentQueries.length === 0 ? (
          <div className="bg-white border border-slate-200 rounded-xl p-6 text-center">
            <Search className="w-6 h-6 text-slate-300 mx-auto mb-2" />
            <p className="text-sm text-slate-400">No queries yet</p>
          </div>
        ) : (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            {recentQueries.map((q, i) => (
              <Link
                key={q.id}
                href={`/research?session=${q.session_id}`}
                className={clsx(
                  'flex items-center gap-3 px-5 py-3.5 hover:bg-slate-50 transition-colors',
                  i < recentQueries.length - 1 && 'border-b border-slate-100'
                )}
              >
                <Search className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                <p className="text-sm text-slate-700 truncate flex-1">
                  {q.content}
                </p>
                <span className="text-[10px] text-slate-400 shrink-0">
                  {new Date(q.created_at).toLocaleDateString()}
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
