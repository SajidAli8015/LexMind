'use client';

import { useState, useCallback } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, X } from 'lucide-react';
import { ingestDocument, IngestResponse } from '@/lib/api';
import Link from 'next/link';
import clsx from 'clsx';

type UploadState = 'idle' | 'uploading' | 'success' | 'error';

export default function UploadPage() {
  const [state, setState] = useState<UploadState>('idle');
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [docTitle, setDocTitle] = useState('');
  const [detectedTitle, setDetectedTitle] = useState('');

  const handleFile = useCallback(async (file: File) => {
    const allowed = ['.pdf', '.docx', '.txt'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowed.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Please upload PDF, DOCX, or TXT.`);
      setState('error');
      return;
    }

    setFileName(file.name);
    setState('uploading');
    setError(null);
    setResult(null);
    setProgress(0);

    // Animate progress bar
    const interval = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 15, 85));
    }, 400);

    try {
      const response = await ingestDocument(file, docTitle || undefined);
      setDetectedTitle(response.doc_title || '');
      clearInterval(interval);
      setProgress(100);
      setTimeout(() => {
        setResult(response);
        setState('success');
      }, 300);
    } catch (err) {
      clearInterval(interval);
      setError(err instanceof Error ? err.message : 'Upload failed');
      setState('error');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const reset = () => {
    setState('idle');
    setResult(null);
    setError(null);
    setFileName(null);
    setProgress(0);
    setDocTitle('');
    setDetectedTitle('');
  };

  return (
    <div className="p-8 max-w-2xl">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-slate-900">Upload document</h1>
        <p className="text-sm text-slate-500 mt-1">
          Add a legal document to your knowledge base. Supported formats: PDF, DOCX, TXT.
        </p>
      </div>

      {/* Drop Zone */}
      {(state === 'idle' || state === 'error') && (
        <label
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={clsx(
            'flex flex-col items-center justify-center w-full h-56 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-200',
            isDragging
              ? 'border-blue-400 bg-blue-50'
              : 'border-slate-300 bg-white hover:border-blue-300 hover:bg-slate-50'
          )}
        >
          <div className={clsx(
            'w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-colors',
            isDragging ? 'bg-blue-100' : 'bg-slate-100'
          )}>
            <Upload className={clsx('w-5 h-5', isDragging ? 'text-blue-500' : 'text-slate-400')} />
          </div>
          <p className="text-sm font-medium text-slate-700">Drag and drop your document here</p>
          <p className="text-xs text-slate-400 mt-1">or click to browse files</p>
          <div className="flex gap-2 mt-4">
            {['PDF', 'DOCX', 'TXT'].map(t => (
              <span key={t} className="px-2.5 py-1 bg-slate-100 border border-slate-200 rounded-full text-[10px] font-medium text-slate-500 uppercase tracking-wide">
                {t}
              </span>
            ))}
          </div>
          <input type="file" className="hidden" accept=".pdf,.docx,.txt" onChange={e => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }} />
        </label>
      )}

      {/* Title input — shown after file is selected */}
      {fileName && state === 'idle' && (
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
      )}

      {/* Uploading */}
      {state === 'uploading' && (
        <div className="bg-white border border-slate-200 rounded-2xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center">
              <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-900">Ingesting {fileName}</p>
              <p className="text-xs text-slate-400 mt-0.5">Parse → Chunk → Embed → Store</p>
            </div>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400 mt-2 text-right">{Math.round(progress)}%</p>
        </div>
      )}

      {/* Error */}
      {state === 'error' && error && (
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

      {/* Success */}
      {state === 'success' && result && (
        <div className="bg-white border border-green-200 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 bg-green-50 rounded-xl flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-900">Document ingested</p>
              <p className="text-xs text-slate-400 mt-0.5">{result.file_name}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-slate-500">Title:</span>
                <span className="text-xs font-medium text-slate-700">
                  {detectedTitle || result.doc_title || result.file_name}
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 mb-4">
            {[
              { label: 'Chunks created', value: result.chunks_created, color: 'bg-blue-50 text-blue-700' },
              { label: 'Articles found', value: result.articles_found, color: 'bg-green-50 text-green-700' },
              { label: 'Characters', value: `${(result.total_chars / 1000).toFixed(1)}k`, color: 'bg-purple-50 text-purple-700' },
            ].map(({ label, value, color }) => (
              <div key={label} className={clsx('rounded-xl p-3 text-center', color)}>
                <p className="text-xl font-bold">{value}</p>
                <p className="text-[11px] opacity-70 mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          <div className="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg mb-4">
            <p className="text-[11px] text-slate-400 font-mono truncate">doc_id: {result.doc_id}</p>
          </div>

          <div className="flex gap-3">
            <Link
              href={`/query?doc_id=${result.doc_id}`}
              className="flex-1 py-2 text-center text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Query this document
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
