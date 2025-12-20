import React from 'react';
import { User, Bot, Loader2 } from 'lucide-react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  // Simple markdown-like parsing
  const formatContent = (content: string) => {
    // Split by code blocks first
    const parts = content.split(/(```[\s\S]*?```)/g);

    return parts.map((part, i) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const code = part.slice(3, -3);
        return (
          <pre key={i} className="bg-gray-100 p-3 rounded-lg overflow-x-auto my-2 text-sm">
            <code>{code}</code>
          </pre>
        );
      }

      // Process inline formatting
      return (
        <div key={i} className="markdown-content">
          {part.split('\n').map((line, j) => {
            // Headers
            if (line.startsWith('## ')) {
              return <h2 key={j} className="text-xl font-bold mt-4 mb-2">{line.slice(3)}</h2>;
            }
            if (line.startsWith('### ')) {
              return <h3 key={j} className="text-lg font-semibold mt-3 mb-2">{line.slice(4)}</h3>;
            }

            // Bold
            let formatted = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            // Lists
            if (line.startsWith('- ')) {
              return (
                <div key={j} className="flex items-start gap-2 ml-4 my-1">
                  <span className="text-blue-500 mt-1">•</span>
                  <span dangerouslySetInnerHTML={{ __html: formatted.slice(2) }} />
                </div>
              );
            }

            // Empty lines
            if (line.trim() === '') {
              return <div key={j} className="h-2" />;
            }

            return (
              <p key={j} className="my-1" dangerouslySetInnerHTML={{ __html: formatted }} />
            );
          })}
        </div>
      );
    });
  };

  return (
    <div className={`flex gap-4 p-4 ${isUser ? 'bg-white' : 'bg-gray-50'}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-600' : 'bg-emerald-600'
        }`}
      >
        {isUser ? (
          <User className="w-6 h-6 text-white" />
        ) : (
          <Bot className="w-6 h-6 text-white" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-gray-900">
            {isUser ? 'You' : 'Research Assistant'}
          </span>
          <span className="text-xs text-gray-400">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>

        {message.isLoading ? (
          <div className="flex items-center gap-2 text-gray-500">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Researching...</span>
          </div>
        ) : (
          <div className="text-gray-700 leading-relaxed">
            {formatContent(message.content)}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
