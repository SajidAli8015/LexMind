'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard, MessageSquare, FileText, Upload, Scale
} from 'lucide-react';
import clsx from 'clsx';

const navItems = [
  { href: '/',          label: 'Dashboard', icon: LayoutDashboard },
  { href: '/research',  label: 'Research',  icon: MessageSquare   },
  { href: '/documents', label: 'Documents', icon: FileText        },
  { href: '/upload',    label: 'Upload',    icon: Upload          },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-60 bg-slate-900 text-white flex flex-col z-10">
      <div className="px-5 py-5 border-b border-slate-700/60">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 bg-blue-500 rounded-lg flex items-center justify-center">
            <Scale className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold tracking-tight">LexMind</p>
            <p className="text-[10px] text-slate-400 leading-none mt-0.5">Legal Research AI</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + '/') && href !== '/';
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150',
                active
                  ? 'bg-blue-600 text-white font-medium'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-slate-700/60">
        <p className="text-[10px] text-slate-500">v0.2.0 · Phase 4</p>
      </div>
    </aside>
  );
}
