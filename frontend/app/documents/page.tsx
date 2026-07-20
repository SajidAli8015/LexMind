'use client';

import { useState, useEffect, useCallback } from 'react';
import { Search, Trash2, ExternalLink, FileText, AlertCircle, Loader2 } from 'lucide-react';
import { listDocuments, deleteDocument, DocumentInfo } from '@/lib/api';
import { useRouter } from 'next/navigation';
import clsx from 'clsx';

function DocTypeBadge({ name }: { name: string }) {
  const ext = name.split('.').pop()?.toLowerCase();
  const styles: Record<string, string> = {
    pdf:  'bg-red-50 text-red-600 border-red-100',
    docx: 'bg-blue-50 text-blue-600 border-blue-100',
    txt:  'bg-slate-50 text-slate-600 border-slate-200',
  };
  return (
    <span className={clsx(
      'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wide',
      styles[ext || ''] || styles.txt
    )}>
      {ext}
    </span>
  );
}

export default function DocumentsPage() {
  const router = useRouter();
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [filtered, setFiltered] = useState<DocumentInfo[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [totalChunks, setTotalChunks] = useState(0);

  const fetchDocs = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocs(data.documents);
      setFiltered(data.documents);
      setTotalChunks(data.total_chunks);
    } catch {
      setError('Failed to load documents. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  useEffect(() => {
    const q = search.toLowerCase();
    setFiltered(docs.filter(d =>
      d.file_name.toLowerCase().includes(q) ||
      d.doc_id.toLowerCase().includes(q)
    ));
  }, [search, docs]);

  const handleDelete = async (docId: string, fileName: string) => {
    if (!confirm(`Delete "${fileName}"? This cannot be undone.`)) return;
    setDeleting(docId);
    try {
      await deleteDocument(docId);
      await fetchDocs();
    } catch {
      setError('Failed to delete document.');
    } finally {
      setDeleting(null);
    }
  };

  const handleQuery = (docId: string) => {
    router.push(`/query?doc_id=${docId}`);
  };

  return (
    <div className="p-8 max-w-5xl">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-900">Documents</h1>
        <p className="text-sm text-slate-500 mt-1">
          {loading ? 'Loading...' : `${docs.length} documents · ${totalChunks} total chunks`}
        </p>
      </div>

      {error && (
        <div className="mb-6 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Search */}
      <div className="flex items-center gap-2.5 px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg mb-4 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all">
        <Search className="w-4 h-4 text-slate-400 shrink-0" />
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search by filename or doc ID..."
          className="flex-1 text-sm bg-transparent outline-none text-slate-900 placeholder-slate-400"
        />
      </div>

      {loading ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center">
          <Loader2 className="w-6 h-6 text-slate-400 animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-400">Loading documents...</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center">
          <FileText className="w-8 h-8 text-slate-300 mx-auto mb-3" />
          <p className="text-sm font-medium text-slate-600">
            {search ? 'No documents match your search' : 'No documents yet'}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            {search ? 'Try a different search term' : 'Upload a legal document to get started'}
          </p>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-12 gap-4 px-5 py-2.5 bg-slate-50 border-b border-slate-200 text-xs font-medium text-slate-500 uppercase tracking-wide">
            <div className="col-span-5">Document</div>
            <div className="col-span-2 text-right">Chunks</div>
            <div className="col-span-2 text-center">Status</div>
            <div className="col-span-3 text-right">Actions</div>
          </div>

          {/* Rows */}
          {filtered.map((doc, i) => (
            <div
              key={doc.doc_id}
              className={clsx(
                'grid grid-cols-12 gap-4 px-5 py-3.5 items-center',
                i < filtered.length - 1 && 'border-b border-slate-100',
                'hover:bg-slate-50 transition-colors'
              )}
            >
              {/* Name */}
              <div className="col-span-5 flex items-center gap-3 min-w-0">
                <DocTypeBadge name={doc.file_name} />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-slate-900 truncate">{doc.file_name}</p>
                  <p className="text-[11px] text-slate-400 font-mono truncate">{doc.doc_id}</p>
                </div>
              </div>

              {/* Chunks */}
              <div className="col-span-2 text-right">
                <span className="text-sm font-medium text-slate-700">{doc.chunk_count}</span>
              </div>

              {/* Status */}
              <div className="col-span-2 flex justify-center">
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-100">
                  Ready
                </span>
              </div>

              {/* Actions */}
              <div className="col-span-3 flex items-center justify-end gap-2">
                <button
                  onClick={() => handleQuery(doc.doc_id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-600 bg-blue-50 border border-blue-100 rounded-lg hover:bg-blue-100 transition-colors"
                >
                  <ExternalLink className="w-3 h-3" />
                  Query
                </button>
                <button
                  onClick={() => handleDelete(doc.doc_id, doc.file_name)}
                  disabled={deleting === doc.doc_id}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-600 bg-red-50 border border-red-100 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50"
                >
                  {deleting === doc.doc_id ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Trash2 className="w-3 h-3" />
                  )}
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
