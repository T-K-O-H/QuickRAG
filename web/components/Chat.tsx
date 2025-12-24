'use client';

import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Message } from './Message';

interface ChatMessage {
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

interface ChatProps {
  messages: ChatMessage[];
}

export function Chat({ messages }: ChatProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div ref={containerRef} className="h-full overflow-y-auto px-6 py-6">
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.map((message, index) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: index * 0.05 }}
          >
            <Message
              role={message.role}
              content={message.content}
              sources={message.sources}
              isStreaming={message.isStreaming}
            />
          </motion.div>
        ))}
      </div>
    </div>
  );
}

