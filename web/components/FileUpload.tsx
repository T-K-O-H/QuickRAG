'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { X, Upload, FileText, Link2, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

interface FileUploadProps {
  onClose: () => void;
}

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export function FileUpload({ onClose }: FileUploadProps) {
  const [activeTab, setActiveTab] = useState<'file' | 'url' | 'text'>('file');
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [message, setMessage] = useState('');
  const [dragActive, setDragActive] = useState(false);

  // File upload state
  const [files, setFiles] = useState<File[]>([]);
  
  // URL upload state
  const [url, setUrl] = useState('');
  
  // Text upload state
  const [text, setText] = useState('');
  const [textSource, setTextSource] = useState('');

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles((prev) => [...prev, ...droppedFiles]);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const uploadFiles = async () => {
    if (files.length === 0) return;

    setStatus('uploading');
    setMessage('');

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    try {
      const res = await fetch('/api/ingest/files', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setStatus('success');
        setMessage(data.message);
        setFiles([]);
      } else {
        setStatus('error');
        setMessage(data.detail || 'Upload failed');
      }
    } catch (error) {
      setStatus('error');
      setMessage('Upload failed. Please try again.');
    }
  };

  const uploadUrl = async () => {
    if (!url.trim()) return;

    setStatus('uploading');
    setMessage('');

    try {
      const res = await fetch('/api/ingest/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();

      if (res.ok) {
        setStatus('success');
        setMessage(data.message);
        setUrl('');
      } else {
        setStatus('error');
        setMessage(data.detail || 'Failed to ingest URL');
      }
    } catch (error) {
      setStatus('error');
      setMessage('Failed to ingest URL. Please try again.');
    }
  };

  const uploadText = async () => {
    if (!text.trim()) return;

    setStatus('uploading');
    setMessage('');

    const formData = new FormData();
    formData.append('text', text);
    formData.append('source', textSource || 'manual');

    try {
      const res = await fetch('/api/ingest/text', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setStatus('success');
        setMessage(data.message);
        setText('');
        setTextSource('');
      } else {
        setStatus('error');
        setMessage(data.detail || 'Failed to ingest text');
      }
    } catch (error) {
      setStatus('error');
      setMessage('Failed to ingest text. Please try again.');
    }
  };

  const handleSubmit = () => {
    if (activeTab === 'file') uploadFiles();
    else if (activeTab === 'url') uploadUrl();
    else uploadText();
  };

  const isSubmitDisabled =
    status === 'uploading' ||
    (activeTab === 'file' && files.length === 0) ||
    (activeTab === 'url' && !url.trim()) ||
    (activeTab === 'text' && !text.trim());

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="w-full max-w-lg bg-surface border border-border rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h2 className="text-lg font-semibold">Add Documents</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-background transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {[
            { id: 'file', label: 'Upload Files', icon: Upload },
            { id: 'url', label: 'From URL', icon: Link2 },
            { id: 'text', label: 'Paste Text', icon: FileText },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => {
                setActiveTab(id as typeof activeTab);
                setStatus('idle');
                setMessage('');
              }}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm transition-colors ${
                activeTab === id
                  ? 'text-accent border-b-2 border-accent'
                  : 'text-muted hover:text-foreground'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6">
          {activeTab === 'file' && (
            <div>
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                  dragActive
                    ? 'border-accent bg-accent/5'
                    : 'border-border hover:border-accent/50'
                }`}
              >
                <input
                  type="file"
                  multiple
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept=".pdf,.txt,.md,.markdown,.rst"
                />
                <Upload className="w-8 h-8 text-muted mx-auto mb-3" />
                <p className="text-sm text-foreground mb-1">
                  Drag & drop files or click to browse
                </p>
                <p className="text-xs text-muted">
                  Supports PDF, TXT, MD, and other text files
                </p>
              </div>

              {files.length > 0 && (
                <div className="mt-4 space-y-2">
                  {files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-background rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <FileText className="w-4 h-4 text-accent" />
                        <span className="text-sm truncate max-w-[200px]">
                          {file.name}
                        </span>
                        <span className="text-xs text-muted">
                          {(file.size / 1024).toFixed(1)} KB
                        </span>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="p-1 hover:bg-surface rounded transition-colors"
                      >
                        <X className="w-4 h-4 text-muted" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'url' && (
            <div>
              <label className="block text-sm font-medium mb-2">
                Web Page URL
              </label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/page"
                className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-accent focus:outline-none"
              />
              <p className="text-xs text-muted mt-2">
                The content of the web page will be extracted and indexed
              </p>
            </div>
          )}

          {activeTab === 'text' && (
            <div>
              <label className="block text-sm font-medium mb-2">
                Text Content
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your text content here..."
                rows={6}
                className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-accent focus:outline-none resize-none"
              />
              <div className="mt-3">
                <label className="block text-sm font-medium mb-2">
                  Source Name (optional)
                </label>
                <input
                  type="text"
                  value={textSource}
                  onChange={(e) => setTextSource(e.target.value)}
                  placeholder="e.g., Meeting Notes"
                  className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-accent focus:outline-none"
                />
              </div>
            </div>
          )}

          {/* Status Message */}
          {message && (
            <div
              className={`mt-4 p-3 rounded-lg flex items-center gap-2 ${
                status === 'success'
                  ? 'bg-green-500/10 text-green-400'
                  : status === 'error'
                  ? 'bg-red-500/10 text-red-400'
                  : ''
              }`}
            >
              {status === 'success' ? (
                <CheckCircle className="w-4 h-4" />
              ) : (
                <AlertCircle className="w-4 h-4" />
              )}
              <span className="text-sm">{message}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-border bg-background/50">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg hover:bg-surface transition-colors text-muted"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitDisabled}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-background font-medium"
          >
            {status === 'uploading' ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Add to Knowledge Base
              </>
            )}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

