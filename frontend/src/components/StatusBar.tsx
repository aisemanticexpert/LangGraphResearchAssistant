import React from 'react';
import { CheckCircle, XCircle, AlertCircle, Database, Clock, Building2 } from 'lucide-react';

interface StatusBarProps {
  isConnected: boolean;
  currentCompany: string | null;
  researchAttempts: number;
  confidenceScore: number;
  cached: boolean;
}

const StatusBar: React.FC<StatusBarProps> = ({
  isConnected,
  currentCompany,
  researchAttempts,
  confidenceScore,
  cached,
}) => {
  return (
    <div className="bg-white border-b px-4 py-2 flex items-center justify-between text-sm">
      {/* Left side - Connection status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <>
              <CheckCircle className="w-4 h-4 text-green-500" />
              <span className="text-green-600">Connected</span>
            </>
          ) : (
            <>
              <XCircle className="w-4 h-4 text-red-500" />
              <span className="text-red-600">Disconnected</span>
            </>
          )}
        </div>

        {currentCompany && (
          <div className="flex items-center gap-2 text-gray-600">
            <Building2 className="w-4 h-4" />
            <span>{currentCompany}</span>
          </div>
        )}
      </div>

      {/* Right side - Stats */}
      <div className="flex items-center gap-4">
        {researchAttempts > 0 && (
          <div className="flex items-center gap-1 text-gray-600">
            <Clock className="w-4 h-4" />
            <span>Attempts: {researchAttempts}/3</span>
          </div>
        )}

        {confidenceScore > 0 && (
          <div className="flex items-center gap-1">
            <span
              className={`font-medium ${
                confidenceScore >= 7
                  ? 'text-green-600'
                  : confidenceScore >= 5
                  ? 'text-yellow-600'
                  : 'text-red-600'
              }`}
            >
              Confidence: {confidenceScore.toFixed(1)}/10
            </span>
          </div>
        )}

        {cached && (
          <div className="flex items-center gap-1 text-blue-600">
            <Database className="w-4 h-4" />
            <span>Cached</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default StatusBar;
