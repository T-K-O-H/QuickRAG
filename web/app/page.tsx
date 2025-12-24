'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Upload, Sparkles, FileText, Link2, Trash2, Zap, Database } from 'lucide-react';
import Link from 'next/link';
import { Chat } from '@/components/Chat';
import { FileUpload } from '@/components/FileUpload';
import { Sources } from '@/components/Sources';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

interface Source {
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [documentCount, setDocumentCount] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Fetch document count on load
  useEffect(() => {
    fetchDocumentCount();
  }, []);

  const fetchDocumentCount = async () => {
    try {
      const res = await fetch('/api/collections');
      if (res.ok) {
        const data = await res.json();
        const current = data.collections.find((c: any) => c.name === data.current);
        setDocumentCount(current?.document_count || 0);
      }
    } catch (e) {
      console.error('Failed to fetch document count:', e);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Create assistant message placeholder
    const assistantId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', isStreaming: true },
    ]);

    try {
      const res = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content, stream: true }),
      });

      if (!res.ok) throw new Error('Query failed');

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let sources: Source[] = [];

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value);
          const lines = text.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'sources') {
                  sources = data.data;
                } else if (data.type === 'chunk') {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? { ...m, content: m.content + data.data }
                        : m
                    )
                  );
                } else if (data.type === 'done') {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? { ...m, isStreaming: false, sources }
                        : m
                    )
                  );
                }
              } catch (e) {
                // Ignore parse errors
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: 'Sorry, an error occurred. Please try again.',
                isStreaming: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-surface/50 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">QuickRAG</h1>
            <p className="text-xs text-muted">Powered by LangGraph</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface border border-border text-sm">
            <FileText className="w-4 h-4 text-accent" />
            <span className="text-muted">{documentCount} chunks</span>
          </div>
          
          <button
            onClick={() => setShowUpload(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface hover:bg-surface-hover border border-border transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span className="text-sm">Upload</span>
          </button>

          <Link
            href="/knowledge"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface hover:bg-surface-hover border border-border transition-colors"
          >
            <Database className="w-4 h-4" />
            <span className="text-sm">Knowledge Base</span>
          </Link>
          
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="p-2 rounded-lg hover:bg-surface-hover transition-colors text-muted hover:text-foreground"
              title="Clear chat"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-hidden">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center px-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center max-w-md"
            >
              <div className="inline-flex p-4 rounded-2xl bg-gradient-to-br from-cyan-500/10 to-blue-600/10 border border-cyan-500/20 mb-6">
                <Sparkles className="w-8 h-8 text-accent" />
              </div>
              <h2 className="text-2xl font-semibold mb-3">Ask anything about your documents</h2>
              <p className="text-muted mb-8">
                Upload documents and ask questions. QuickRAG uses hybrid search 
                to find the most relevant information.
              </p>
              
              <div className="grid grid-cols-2 gap-3 text-sm">
                <button
                  onClick={() => setShowUpload(true)}
                  className="flex items-center gap-2 p-4 rounded-xl bg-surface hover:bg-surface-hover border border-border transition-all hover:border-accent/50"
                >
                  <Upload className="w-5 h-5 text-accent" />
                  <span>Upload files</span>
                </button>
                <button
                  onClick={() => setInput('What can you tell me about...')}
                  className="flex items-center gap-2 p-4 rounded-xl bg-surface hover:bg-surface-hover border border-border transition-all hover:border-accent/50"
                >
                  <Link2 className="w-5 h-5 text-accent" />
                  <span>Ask a question</span>
                </button>
              </div>
            </motion.div>
          </div>
        ) : (
          <Chat messages={messages} />
        )}
      </main>

      {/* Input Area */}
      <div className="border-t border-border bg-surface/50 backdrop-blur-sm p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="relative flex items-end gap-2">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question..."
                rows={1}
                className="w-full px-4 py-3 pr-12 rounded-xl bg-background border border-border focus:border-accent focus:outline-none resize-none text-foreground placeholder:text-muted transition-colors"
                style={{ minHeight: '48px', maxHeight: '200px' }}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="absolute right-2 bottom-2 p-2 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <Send className="w-4 h-4 text-background" />
              </button>
            </div>
          </div>
          <p className="text-xs text-muted text-center mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </form>
      </div>

      {/* Upload Modal */}
      <AnimatePresence>
        {showUpload && (
          <FileUpload
            onClose={() => {
              setShowUpload(false);
              fetchDocumentCount();
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

