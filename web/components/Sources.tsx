'use client';

import { FileText, ExternalLink } from 'lucide-react';

interface Source {
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

interface SourcesProps {
  sources: Source[];
}

export function Sources({ sources }: SourcesProps) {
  return (
    <div className="space-y-2">
      {sources.map((source, index) => {
        const filename = source.metadata.original_filename || 
                        source.metadata.filename ||
                        source.metadata.source ||
                        'Unknown source';
        const relevance = Math.round(source.score * 100);

        return (
          <div
            key={index}
            className="p-3 rounded-xl bg-background border border-border hover:border-accent/30 transition-colors"
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 p-1.5 rounded-lg bg-surface">
                <FileText className="w-4 h-4 text-accent" />
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium truncate">
                    {String(filename)}
                  </span>
                  <span className="flex-shrink-0 px-2 py-0.5 rounded-full text-xs bg-accent/10 text-accent">
                    {relevance}% match
                  </span>
                </div>
                
                <p className="text-xs text-muted line-clamp-3">
                  {source.content.slice(0, 200)}
                  {source.content.length > 200 ? '...' : ''}
                </p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

