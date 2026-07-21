'use client';

import { useCallback } from 'react';
import {
  Upload, FileText, CheckCircle, AlertCircle,
  Loader2, X, ArrowRight
} from 'lucide-react';
import { useUpload } from '@/contexts/UploadContext';
import Link from 'next/link';
import clsx from 'clsx';

export default function UploadPage() {
  const {
    status, fileName, progress, result, error,
    docTitle, startUpload, reset, setDocTitle
  } = useUpload();

  const handleFile = useCallback((file: File) => {
    startUpload(file, docTitle || undefined);
  }, [startUpload, docTitle]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="p-8 max-w-2xl">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-slate-900">Upload document</h1>
        <p className="text-sm text-slate-500 mt-1">
          Add a legal document to your knowledge base.
          Supported formats: PDF, DOCX, TXT.
        </p>
      </div>

      {/* Drop Zone — shown when idle or error */}
      {(status === 'idle' || status === 'error') && (
        <>
          <label
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
            className="flex flex-col items-center justify-center w-full h-52 border-2 border-dashed border-slate-300 rounded-2xl cursor-pointer bg-white hover:border-blue-300 hover:bg-slate-50 transition-all duration-200"
          >
            <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center mb-4">
              <Upload className="w-5 h-5 text-slate-400" />
            </div>
            <p className="text-sm font-medium text-slate-700">
              Drag and drop your document here
            </p>
            <p className="text-xs text-slate-400 mt-1">or click to browse files</p>
            <div className="flex gap-2 mt-4">
              {['PDF', 'DOCX', 'TXT'].map(t => (
                <span key={t} className="px-2.5 py-1 bg-slate-100 border border-slate-200 rounded-full text-[10px] font-medium text-slate-500 uppercase tracking-wide">
                  {t}
                </span>
              ))}
            </div>
            <input
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt"
              onChange={e => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
          </label>

          {/* Optional title input */}
          <div className="mt-4">
            <label className="block text-xs font-medium text-slate-600 mb-1.5">
              Document title
              <span className="ml-1 text-slate-400 font-normal">
                (optional — auto-detected if left empty)
              </span>
            </label>
            <input
              value={docTitle}
              onChange={e => setDocTitle(e.target.value)}
              placeholder="e.g. Saudi Labor Law, NDA Agreement 2024..."
              className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors"
            />
          </div>
        </>
      )}

      {/* Error message */}
      {status === 'error' && error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-red-700">Upload failed</p>
            <p className="text-sm text-red-600 mt-0.5">{error}</p>
          </div>
          <button onClick={reset} className="text-red-400 hover:text-red-600">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Uploading state */}
      {status === 'uploading' && (
        <div className="bg-white border border-slate-200 rounded-2xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center">
              <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-900">
                Ingesting {fileName}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                Parse → Chunk → Embed → Store
              </p>
            </div>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400 mt-2 text-right">
            {Math.round(progress)}% — you can navigate away safely
          </p>
        </div>
      )}

      {/* Success state */}
      {status === 'success' && result && (
        <div className="bg-white border border-green-200 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 bg-green-50 rounded-xl flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-900">
                Document ingested
              </p>
              <p className="text-xs text-slate-400 mt-0.5">{result.file_name}</p>
              {result.doc_title && (
                <p className="text-xs text-blue-600 font-medium mt-0.5">
                  {result.doc_title}
                </p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 mb-4">
            {[
              { label: 'Chunks', value: result.chunks_created, color: 'bg-blue-50 text-blue-700' },
              { label: 'Articles', value: result.articles_found, color: 'bg-green-50 text-green-700' },
              { label: 'Characters', value: `${(result.total_chars / 1000).toFixed(1)}k`, color: 'bg-purple-50 text-purple-700' },
            ].map(({ label, value, color }) => (
              <div key={label} className={clsx('rounded-xl p-3 text-center', color)}>
                <p className="text-xl font-bold">{value}</p>
                <p className="text-[11px] opacity-70 mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          <div className="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg mb-4">
            <p className="text-[11px] text-slate-400 font-mono truncate">
              doc_id: {result.doc_id}
            </p>
          </div>

          <div className="flex gap-3">
            <Link
              href={`/research?doc_id=${result.doc_id}`}
              className="flex-1 py-2 text-center text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
            >
              Query this document
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <button
              onClick={reset}
              className="px-4 py-2 text-sm font-medium text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors"
            >
              Upload another
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
