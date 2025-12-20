import React from 'react';
import { Search, Building2, Brain, RefreshCw, Users } from 'lucide-react';

interface WelcomeScreenProps {
  onQuickQuery: (query: string) => void;
  companies: string[];
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onQuickQuery, companies }) => {
  const quickQueries = [
    "Tell me about Apple's latest products",
    "What is NVIDIA working on with AI?",
    "Compare Tesla and Ford's EV strategy",
    "How is Microsoft using AI?",
    "What's new with Amazon's cloud business?",
    "Tell me about Pfizer's drug pipeline",
  ];

  const features = [
    {
      icon: Brain,
      title: '4 AI Agents',
      description: 'Clarity, Research, Validator, and Synthesis agents work together',
    },
    {
      icon: Building2,
      title: '28 Companies',
      description: 'Comprehensive data on major companies across industries',
    },
    {
      icon: RefreshCw,
      title: 'Smart Retry',
      description: 'Automatic retry logic ensures quality research',
    },
    {
      icon: Users,
      title: 'Human-in-the-Loop',
      description: 'Asks for clarification when queries are unclear',
    },
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-3xl w-full text-center">
        {/* Logo and Title */}
        <div className="mb-8">
          <div className="w-20 h-20 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <Search className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Research Assistant
          </h1>
          <p className="text-xl text-gray-600">
            Multi-Agent AI-Powered Company Research
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white rounded-xl p-4 shadow-sm border border-gray-100"
            >
              <feature.icon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
              <h3 className="font-semibold text-gray-900 text-sm">{feature.title}</h3>
              <p className="text-xs text-gray-500 mt-1">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Quick Queries */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">Try a quick query</h2>
          <div className="flex flex-wrap justify-center gap-2">
            {quickQueries.map((query, index) => (
              <button
                key={index}
                onClick={() => onQuickQuery(query)}
                className="bg-white hover:bg-blue-50 border border-gray-200 hover:border-blue-300 text-gray-700 hover:text-blue-700 px-4 py-2 rounded-full text-sm transition-colors shadow-sm"
              >
                {query}
              </button>
            ))}
          </div>
        </div>

        {/* Company Tags */}
        <div>
          <h2 className="text-sm font-medium text-gray-500 mb-3">
            Available Companies
          </h2>
          <div className="flex flex-wrap justify-center gap-2 max-h-24 overflow-hidden">
            {companies.slice(0, 15).map((company, index) => (
              <span
                key={index}
                onClick={() => onQuickQuery(`Tell me about ${company}`)}
                className="bg-gray-100 hover:bg-gray-200 text-gray-600 px-3 py-1 rounded-full text-xs cursor-pointer transition-colors"
              >
                {company}
              </span>
            ))}
            {companies.length > 15 && (
              <span className="text-gray-400 text-xs px-2 py-1">
                +{companies.length - 15} more
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
