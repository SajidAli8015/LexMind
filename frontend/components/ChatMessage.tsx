'use client';

import { MessageResponse } from '@/lib/api';
import clsx from 'clsx';

const QUERY_TYPE_STYLES: Record<string, string> = {
  factual:       'bg-blue-50 text-blue-700 border-blue-100',
  analytical:    'bg-purple-50 text-purple-700 border-purple-100',
  comparison:    'bg-amber-50 text-amber-700 border-amber-100',
  summarisation: 'bg-green-50 text-green-700 border-green-100',
};

function ScoreBadge({ label, score }: { label: string; score: number | null | undefined }) {
  if (score === null || score === undefined) return null;
  const pct = Math.round(score * 100);
  const color = pct >= 80
    ? 'bg-green-50 text-green-700 border-green-100'
    : pct >= 60
    ? 'bg-amber-50 text-amber-700 border-amber-100'
    : 'bg-red-50 text-red-700 border-red-100';
  return (
    <span className={clsx('inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10px] font-medium', color)}>
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
          <span key={i} className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-100 text-blue-800 mx-0.5">
            {part}
          </span>
        ) : part
      )}
    </>
  );
}

function formatTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

interface Props {
  message: MessageResponse;
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex flex-col items-end gap-1">
        <div className="max-w-[80%] px-4 py-2.5 bg-blue-600 text-white rounded-2xl rounded-tr-sm text-sm leading-relaxed">
          {message.content}
        </div>
        <span className="text-[10px] text-slate-400 px-1">{formatTime(message.created_at)}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <div className="max-w-[85%] bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 text-sm leading-relaxed text-slate-800">
        <p className="whitespace-pre-wrap">
          {highlightCitations(message.content, message.citations || [])}
        </p>
      </div>

      {/* Citations */}
      {message.citations && message.citations.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap px-1">
          <span className="text-[10px] text-slate-400 font-medium">Cited:</span>
          {message.citations.map(c => (
            <span key={c} className="px-2 py-0.5 bg-blue-50 border border-blue-100 rounded-full text-[10px] font-medium text-blue-700">
              {c}
            </span>
          ))}
        </div>
      )}

      {/* Quality scores + meta */}
      <div className="flex items-center gap-2 flex-wrap px-1">
        {message.query_type && (
          <span className={clsx(
            'px-2 py-0.5 rounded-full text-[10px] font-medium border capitalize',
            QUERY_TYPE_STYLES[message.query_type] || 'bg-slate-50 text-slate-600 border-slate-200'
          )}>
            {message.query_type}
          </span>
        )}
        <ScoreBadge label="Grounded" score={message.groundedness_score} />
        <ScoreBadge label="Citations" score={message.citation_score} />
        <ScoreBadge label="Relevant" score={message.relevance_score} />
        {message.chunks_used !== undefined && (
          <span className="text-[10px] text-slate-400">{message.chunks_used} chunks</span>
        )}
        <span className="text-[10px] text-slate-400">{formatTime(message.created_at)}</span>
      </div>
    </div>
  );
}
