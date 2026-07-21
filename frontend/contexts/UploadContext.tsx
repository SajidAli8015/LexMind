'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { ingestDocument, IngestResponse } from '@/lib/api';

export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

interface UploadState {
  status: UploadStatus;
  fileName: string | null;
  progress: number;
  result: IngestResponse | null;
  error: string | null;
  docTitle: string;
}

interface UploadContextType extends UploadState {
  startUpload: (file: File, docTitle?: string) => Promise<void>;
  reset: () => void;
  setDocTitle: (title: string) => void;
}

const UploadContext = createContext<UploadContextType | null>(null);

const INITIAL_STATE: UploadState = {
  status: 'idle',
  fileName: null,
  progress: 0,
  result: null,
  error: null,
  docTitle: '',
};

export function UploadProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<UploadState>(INITIAL_STATE);

  const setDocTitle = useCallback((title: string) => {
    setState(prev => ({ ...prev, docTitle: title }));
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  const startUpload = useCallback(async (file: File, docTitle?: string) => {
    const allowed = ['.pdf', '.docx', '.txt'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();

    if (!allowed.includes(ext)) {
      setState(prev => ({
        ...prev,
        status: 'error',
        error: `Unsupported file type: ${ext}. Please upload PDF, DOCX, or TXT.`,
      }));
      return;
    }

    setState({
      status: 'uploading',
      fileName: file.name,
      progress: 0,
      result: null,
      error: null,
      docTitle: docTitle || '',
    });

    // Animate progress bar
    let progressInterval: ReturnType<typeof setInterval>;
    progressInterval = setInterval(() => {
      setState(prev => ({
        ...prev,
        progress: Math.min(prev.progress + Math.random() * 12, 85),
      }));
    }, 400);

    try {
      const response = await ingestDocument(file, docTitle);
      clearInterval(progressInterval);

      setState(prev => ({
        ...prev,
        status: 'success',
        progress: 100,
        result: response,
      }));
    } catch (err) {
      clearInterval(progressInterval);
      setState(prev => ({
        ...prev,
        status: 'error',
        error: err instanceof Error ? err.message : 'Upload failed',
        progress: 0,
      }));
    }
  }, []);

  return (
    <UploadContext.Provider value={{ ...state, startUpload, reset, setDocTitle }}>
      {children}
    </UploadContext.Provider>
  );
}

export function useUpload() {
  const context = useContext(UploadContext);
  if (!context) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
}
