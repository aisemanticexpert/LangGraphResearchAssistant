import React, { useState } from 'react';
import { HelpCircle, Send, X } from 'lucide-react';

interface ClarificationModalProps {
  question: string;
  originalQuery: string;
  onSubmit: (clarification: string) => void;
  onCancel: () => void;
}

const ClarificationModal: React.FC<ClarificationModalProps> = ({
  question,
  originalQuery,
  onSubmit,
  onCancel,
}) => {
  const [clarification, setClarification] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (clarification.trim()) {
      onSubmit(clarification.trim());
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-md w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <HelpCircle className="w-6 h-6 text-amber-500" />
            <h2 className="text-lg font-semibold">Clarification Needed</h2>
          </div>
          <button
            onClick={onCancel}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-600 mb-1">Original query:</p>
            <p className="text-gray-800 italic">"{originalQuery}"</p>
          </div>

          <p className="text-gray-700 mb-4">{question}</p>

          <form onSubmit={handleSubmit}>
            <textarea
              value={clarification}
              onChange={(e) => setClarification(e.target.value)}
              placeholder="Provide more details..."
              className="w-full border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              rows={3}
              autoFocus
            />

            <div className="flex justify-end gap-3 mt-4">
              <button
                type="button"
                onClick={onCancel}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!clarification.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ClarificationModal;
