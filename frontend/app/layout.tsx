import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { UploadProvider } from '@/contexts/UploadContext';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'LexMind — Legal Research Assistant',
  description: 'AI-powered multi-agent legal document research',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className} style={{backgroundColor: '#f8fafc', color: '#0f172a'}}>
        <UploadProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <main className="flex-1 ml-60 min-h-screen" style={{backgroundColor: '#f8fafc'}}>
              {children}
            </main>
          </div>
        </UploadProvider>
      </body>
    </html>
  );
}
