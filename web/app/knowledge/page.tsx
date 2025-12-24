'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
  Database,
  FileText,
  Trash2,
  Plus,
  ArrowLeft,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Upload,
  Link2,
  Loader2,
  FolderOpen,
  Globe,
  File,
  Type,
} from 'lucide-react';

interface Collection {
  name: string;
  document_count: number;
  embedding_dim: number;
}

interface Document {
  document_id: string;
  source: string;
  source_type: string;
  filename: string | null;
  chunk_count: number;
  created_at: string | null;
}

export default function KnowledgePage() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [currentCollection, setCurrentCollection] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'documents' | 'collections'>('documents');
  
  // Upload states
  const [showUpload, setShowUpload] = useState(false);
  const [uploadType, setUploadType] = useState<'file' | 'url' | 'text'>('file');
  const [files, setFiles] = useState<File[]>([]);
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  const [textSource, setTextSource] = useState('');
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [collectionsRes, documentsRes] = await Promise.all([
        fetch('/api/collections'),
        fetch('/api/documents'),
      ]);
      
      if (collectionsRes.ok) {
        const data = await collectionsRes.json();
        setCollections(data.collections);
        setCurrentCollection(data.current);
      }
      
      if (documentsRes.ok) {
        const data = await documentsRes.json();
        setDocuments(data.documents);
      }
    } catch (e) {
      console.error('Failed to fetch data:', e);
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (documentId: string, source: string) => {
    if (!confirm(`Delete "${source}" and all its chunks?`)) {
      return;
    }

    setActionLoading(documentId);
    try {
      const res = await fetch(`/api/documents/${documentId}`, { method: 'DELETE' });
      if (res.ok) {
        const data = await res.json();
        setMessage({ type: 'success', text: data.message });
        fetchData();
      } else {
        const data = await res.json();
        setMessage({ type: 'error', text: data.detail || 'Failed to delete document' });
      }
    } catch (e) {
      setMessage({ type: 'error', text: 'Failed to delete document' });
    } finally {
      setActionLoading(null);
    }
  };

  const clearCollection = async (name: string) => {
    if (!confirm(`Clear ALL documents from "${name}"? This cannot be undone.`)) {
      return;
    }

    setActionLoading(name);
    try {
      const res = await fetch(`/api/collections/${name}/clear`, { method: 'POST' });
      if (res.ok) {
        setMessage({ type: 'success', text: `Cleared collection: ${name}` });
        fetchData();
      } else {
        const data = await res.json();
        setMessage({ type: 'error', text: data.detail || 'Failed to clear collection' });
      }
    } catch (e) {
      setMessage({ type: 'error', text: 'Failed to clear collection' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    setMessage(null);

    try {
      let res;
      
      if (uploadType === 'file' && files.length > 0) {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        res = await fetch('/api/ingest/files', { method: 'POST', body: formData });
      } else if (uploadType === 'url' && url.trim()) {
        res = await fetch('/api/ingest/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
      } else if (uploadType === 'text' && text.trim()) {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('source', textSource || 'Manual Entry');
        res = await fetch('/api/ingest/text', { method: 'POST', body: formData });
      } else {
        setMessage({ type: 'error', text: 'Please provide content to upload' });
        setUploading(false);
        return;
      }

      if (res && res.ok) {
        const data = await res.json();
        setMessage({ type: 'success', text: data.message });
        setFiles([]);
        setUrl('');
        setText('');
        setTextSource('');
        setShowUpload(false);
        fetchData();
      } else if (res) {
        const data = await res.json();
        setMessage({ type: 'error', text: data.detail || 'Upload failed' });
      }
    } catch (e) {
      setMessage({ type: 'error', text: 'Upload failed' });
    } finally {
      setUploading(false);
    }
  };

  const getSourceIcon = (sourceType: string) => {
    switch (sourceType) {
      case 'url':
        return <Globe className="w-4 h-4 text-blue-400" />;
      case 'file':
        return <File className="w-4 h-4 text-green-400" />;
      case 'text':
        return <Type className="w-4 h-4 text-purple-400" />;
      default:
        return <FileText className="w-4 h-4 text-muted" />;
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Unknown';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return 'Unknown';
    }
  };

  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-surface/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="p-2 rounded-lg hover:bg-background transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600">
                  <Database className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold">Knowledge Base</h1>
                  <p className="text-xs text-muted">Manage your documents and collections</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={fetchData}
                disabled={loading}
                className="p-2 rounded-lg hover:bg-background transition-colors text-muted hover:text-foreground"
              >
                <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setShowUpload(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover transition-colors text-background font-medium"
              >
                <Plus className="w-4 h-4" />
                Add Documents
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* Message */}
        <AnimatePresence>
          {message && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className={`mb-6 p-4 rounded-xl flex items-center gap-3 ${
                message.type === 'success'
                  ? 'bg-green-500/10 border border-green-500/20 text-green-400'
                  : 'bg-red-500/10 border border-red-500/20 text-red-400'
              }`}
            >
              {message.type === 'success' ? (
                <CheckCircle className="w-5 h-5" />
              ) : (
                <AlertCircle className="w-5 h-5" />
              )}
              <span>{message.text}</span>
              <button
                onClick={() => setMessage(null)}
                className="ml-auto p-1 hover:bg-white/10 rounded"
              >
                ×
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="p-4 rounded-xl bg-surface border border-border">
            <div className="text-2xl font-bold text-accent">{documents.length}</div>
            <div className="text-sm text-muted">Documents</div>
          </div>
          <div className="p-4 rounded-xl bg-surface border border-border">
            <div className="text-2xl font-bold text-accent">{totalChunks}</div>
            <div className="text-sm text-muted">Total Chunks</div>
          </div>
          <div className="p-4 rounded-xl bg-surface border border-border">
            <div className="text-2xl font-bold text-green-400">Active</div>
            <div className="text-sm text-muted">Collection: {currentCollection}</div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab('documents')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'documents'
                ? 'bg-accent text-background'
                : 'bg-surface hover:bg-surface-hover text-muted'
            }`}
          >
            Documents ({documents.length})
          </button>
          <button
            onClick={() => setActiveTab('collections')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'collections'
                ? 'bg-accent text-background'
                : 'bg-surface hover:bg-surface-hover text-muted'
            }`}
          >
            Collections ({collections.length})
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-accent" />
          </div>
        ) : activeTab === 'documents' ? (
          /* Documents List */
          documents.length === 0 ? (
            <div className="text-center py-12">
              <FolderOpen className="w-12 h-12 text-muted mx-auto mb-4" />
              <p className="text-muted">No documents found</p>
              <button
                onClick={() => setShowUpload(true)}
                className="mt-4 text-accent hover:underline"
              >
                Upload your first document
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {documents.map((doc) => (
                <motion.div
                  key={doc.document_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-xl bg-surface border border-border hover:border-border/80 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 min-w-0 flex-1">
                      <div className="p-2 rounded-lg bg-background flex-shrink-0">
                        {getSourceIcon(doc.source_type)}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium truncate" title={doc.source}>
                            {doc.source}
                          </h3>
                          <span className="flex-shrink-0 px-2 py-0.5 rounded-full text-xs bg-accent/10 text-accent">
                            {doc.chunk_count} chunks
                          </span>
                        </div>
                        <p className="text-sm text-muted">
                          {doc.source_type === 'file' && 'File'}
                          {doc.source_type === 'url' && 'URL'}
                          {doc.source_type === 'text' && 'Text'}
                          {' · '}
                          {formatDate(doc.created_at)}
                        </p>
                      </div>
                    </div>

                    <button
                      onClick={() => deleteDocument(doc.document_id, doc.source)}
                      disabled={actionLoading === doc.document_id}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm hover:bg-background transition-colors text-red-400 flex-shrink-0 ml-4"
                    >
                      {actionLoading === doc.document_id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                      Delete
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          )
        ) : (
          /* Collections List */
          <div className="space-y-3">
            {collections.map((collection) => (
              <motion.div
                key={collection.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 rounded-xl bg-surface border transition-colors ${
                  collection.name === currentCollection
                    ? 'border-accent/50'
                    : 'border-border hover:border-border/80'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="p-2 rounded-lg bg-background">
                      <Database className="w-5 h-5 text-accent" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium">{collection.name}</h3>
                        {collection.name === currentCollection && (
                          <span className="px-2 py-0.5 rounded-full text-xs bg-accent/10 text-accent">
                            Active
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-muted">
                        {collection.document_count} chunks
                        {collection.embedding_dim > 0 && ` · ${collection.embedding_dim}d embeddings`}
                      </p>
                    </div>
                  </div>

                  {collection.name === currentCollection && collection.document_count > 0 && (
                    <button
                      onClick={() => clearCollection(collection.name)}
                      disabled={actionLoading === collection.name}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm hover:bg-background transition-colors text-amber-400"
                    >
                      {actionLoading === collection.name ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                      Clear All
                    </button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </main>

      {/* Upload Modal */}
      <AnimatePresence>
        {showUpload && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={() => setShowUpload(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-lg bg-surface border border-border rounded-2xl shadow-2xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-6 py-4 border-b border-border">
                <h2 className="text-lg font-semibold">Add Document</h2>
                <button
                  onClick={() => setShowUpload(false)}
                  className="p-1.5 rounded-lg hover:bg-background transition-colors text-xl"
                >
                  ×
                </button>
              </div>

              {/* Tabs */}
              <div className="flex border-b border-border">
                {[
                  { id: 'file', label: 'Upload File', icon: Upload },
                  { id: 'url', label: 'From URL', icon: Globe },
                  { id: 'text', label: 'Paste Text', icon: Type },
                ].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setUploadType(id as typeof uploadType)}
                    className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm transition-colors ${
                      uploadType === id
                        ? 'text-accent border-b-2 border-accent'
                        : 'text-muted hover:text-foreground'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </button>
                ))}
              </div>

              <div className="p-6">
                {uploadType === 'file' && (
                  <div>
                    <div className="relative border-2 border-dashed border-border rounded-xl p-8 text-center hover:border-accent/50 transition-colors cursor-pointer">
                      <input
                        type="file"
                        multiple
                        onChange={(e) => setFiles(Array.from(e.target.files || []))}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                      <Upload className="w-8 h-8 text-muted mx-auto mb-3" />
                      <p className="text-sm">Click to select files</p>
                      <p className="text-xs text-muted mt-1">PDF, TXT, MD supported</p>
                    </div>
                    {files.length > 0 && (
                      <div className="mt-4 space-y-2">
                        {files.map((file, i) => (
                          <div key={i} className="flex items-center gap-2 text-sm p-2 bg-background rounded-lg">
                            <File className="w-4 h-4 text-green-400" />
                            <span className="truncate">{file.name}</span>
                            <span className="text-muted text-xs">({(file.size / 1024).toFixed(1)} KB)</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {uploadType === 'url' && (
                  <div>
                    <label className="block text-sm font-medium mb-2">Web Page URL</label>
                    <input
                      type="url"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="https://example.com/page"
                      className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-accent focus:outline-none"
                    />
                    <p className="text-xs text-muted mt-2">
                      The content will be extracted and stored as a separate document
                    </p>
                  </div>
                )}

                {uploadType === 'text' && (
                  <div>
                    <label className="block text-sm font-medium mb-2">Document Name</label>
                    <input
                      type="text"
                      value={textSource}
                      onChange={(e) => setTextSource(e.target.value)}
                      placeholder="e.g., Meeting Notes, Product Info"
                      className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-accent focus:outline-none mb-4"
                    />
                    <label className="block text-sm font-medium mb-2">Content</label>
                    <textarea
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      placeholder="Paste your text here..."
                      rows={6}
                      className="w-full px-4 py-3 rounded-xl bg-background border border-border focus:border-accent focus:outline-none resize-none"
                    />
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-3 px-6 py-4 border-t border-border bg-background/50">
                <button
                  onClick={() => setShowUpload(false)}
                  className="px-4 py-2 rounded-lg hover:bg-surface transition-colors text-muted"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-50 transition-colors text-background font-medium"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4" />
                      Add Document
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
