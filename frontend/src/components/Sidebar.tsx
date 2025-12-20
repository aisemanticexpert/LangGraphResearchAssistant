import React from 'react';
import { Building2, MessageSquare, Trash2, Plus, Database } from 'lucide-react';
import { Conversation } from '../types';

interface SidebarProps {
  companies: string[];
  conversations: Conversation[];
  currentThreadId: string | null;
  onNewChat: () => void;
  onSelectConversation: (threadId: string) => void;
  onDeleteConversation: (threadId: string) => void;
  onSelectCompany: (company: string) => void;
  cacheStats: { valid_entries: number; max_size: number } | null;
}

const Sidebar: React.FC<SidebarProps> = ({
  companies,
  conversations,
  currentThreadId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onSelectCompany,
  cacheStats,
}) => {
  const [showCompanies, setShowCompanies] = React.useState(false);

  return (
    <div className="w-72 bg-gray-900 text-white flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Building2 className="w-6 h-6 text-blue-400" />
          Research Assistant
        </h1>
        <p className="text-gray-400 text-sm mt-1">Multi-Agent AI Research</p>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Research
        </button>
      </div>

      {/* Companies Section */}
      <div className="px-4 mb-2">
        <button
          onClick={() => setShowCompanies(!showCompanies)}
          className="w-full flex items-center justify-between text-gray-400 hover:text-white py-2 text-sm"
        >
          <span className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            Companies ({companies.length})
          </span>
          <span>{showCompanies ? '−' : '+'}</span>
        </button>

        {showCompanies && (
          <div className="max-h-48 overflow-y-auto bg-gray-800 rounded-lg p-2 mb-2">
            {companies.map((company) => (
              <button
                key={company}
                onClick={() => onSelectCompany(company)}
                className="w-full text-left text-sm text-gray-300 hover:text-white hover:bg-gray-700 px-2 py-1 rounded transition-colors"
              >
                {company}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Conversations */}
      <div className="flex-1 overflow-y-auto px-4">
        <h3 className="text-gray-400 text-sm font-medium mb-2 flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Recent Conversations
        </h3>

        {conversations.length === 0 ? (
          <p className="text-gray-500 text-sm italic">No conversations yet</p>
        ) : (
          <div className="space-y-1">
            {conversations.map((conv) => (
              <div
                key={conv.threadId}
                className={`group flex items-center justify-between rounded-lg p-2 cursor-pointer transition-colors ${
                  currentThreadId === conv.threadId
                    ? 'bg-gray-700'
                    : 'hover:bg-gray-800'
                }`}
                onClick={() => onSelectConversation(conv.threadId)}
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">
                    {conv.company || 'Research Query'}
                  </p>
                  <p className="text-xs text-gray-400">
                    {new Date(conv.createdAt).toLocaleDateString()}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.threadId);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-600 rounded transition-opacity"
                >
                  <Trash2 className="w-4 h-4 text-gray-400 hover:text-red-400" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Cache Stats */}
      {cacheStats && (
        <div className="p-4 border-t border-gray-700">
          <div className="text-xs text-gray-400">
            Cache: {cacheStats.valid_entries}/{cacheStats.max_size} entries
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
