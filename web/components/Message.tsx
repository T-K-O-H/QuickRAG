'use client';

import { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sources } from './Sources';

interface Source {
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

export function Message({ role, content, sources, isStreaming }: MessageProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = role === 'user';

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
          isUser
            ? 'bg-gradient-to-br from-violet-500 to-purple-600'
            : 'bg-gradient-to-br from-cyan-500 to-blue-600'
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-gradient-to-br from-violet-500/20 to-purple-600/20 border border-violet-500/30'
              : 'bg-surface border border-border'
          }`}
        >
          {content ? (
            <div className={`prose text-foreground ${isUser ? 'text-right' : 'text-left'}`}>
              {content.split('\n').map((line, i) => (
                <p key={i} className="mb-2 last:mb-0">
                  {line}
                </p>
              ))}
            </div>
          ) : isStreaming ? (
            <div className="flex items-center gap-1 py-1">
              <span className="typing-dot w-2 h-2 rounded-full bg-accent"></span>
              <span className="typing-dot w-2 h-2 rounded-full bg-accent"></span>
              <span className="typing-dot w-2 h-2 rounded-full bg-accent"></span>
            </div>
          ) : null}
        </div>

        {/* Sources Toggle */}
        {sources && sources.length > 0 && !isStreaming && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="inline-flex items-center gap-1.5 text-xs text-muted hover:text-foreground transition-colors"
            >
              <FileText className="w-3 h-3" />
              <span>{sources.length} source{sources.length > 1 ? 's' : ''}</span>
              {showSources ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </button>

            <AnimatePresence>
              {showSources && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                  className="mt-2 overflow-hidden"
                >
                  <Sources sources={sources} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}

