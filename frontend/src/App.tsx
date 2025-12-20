import React, { useState, useEffect, useRef } from 'react';
import './index.css';
import researchApi from './services/api';
import { Message, Conversation, CacheStats, InterruptInfo } from './types';
import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import StatusBar from './components/StatusBar';
import WelcomeScreen from './components/WelcomeScreen';
import ClarificationModal from './components/ClarificationModal';

function App() {
  // State
  const [companies, setCompanies] = useState<string[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [interruptInfo, setInterruptInfo] = useState<InterruptInfo | null>(null);
  const [currentState, setCurrentState] = useState<{
    company: string | null;
    attempts: number;
    confidence: number;
    cached: boolean;
  }>({ company: null, attempts: 0, confidence: 0, cached: false });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentConversation?.messages]);

  // Initialize - fetch companies and check health
  useEffect(() => {
    const init = async () => {
      try {
        const [healthResponse, companiesResponse] = await Promise.all([
          researchApi.getHealth(),
          researchApi.getCompanies(),
        ]);
        setIsConnected(healthResponse.status === 'healthy');
        setCacheStats(healthResponse.cache_stats);
        setCompanies(companiesResponse.companies);
      } catch (error) {
        console.error('Failed to connect to API:', error);
        setIsConnected(false);
      }
    };
    init();
  }, []);

  // Generate unique ID
  const generateId = () => Math.random().toString(36).substring(2, 15);

  // Create new chat
  const handleNewChat = () => {
    setCurrentConversation(null);
    setCurrentState({ company: null, attempts: 0, confidence: 0, cached: false });
    setInterruptInfo(null);
  };

  // Select existing conversation
  const handleSelectConversation = (threadId: string) => {
    const conv = conversations.find((c) => c.threadId === threadId);
    if (conv) {
      setCurrentConversation(conv);
      setCurrentState({
        company: conv.company,
        attempts: 0,
        confidence: 0,
        cached: false,
      });
    }
  };

  // Delete conversation
  const handleDeleteConversation = (threadId: string) => {
    setConversations((prev) => prev.filter((c) => c.threadId !== threadId));
    if (currentConversation?.threadId === threadId) {
      setCurrentConversation(null);
    }
  };

  // Select company (quick query)
  const handleSelectCompany = (company: string) => {
    handleSendMessage(`Tell me about ${company}`);
  };

  // Send message
  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    setIsLoading(true);
    setInterruptInfo(null);

    // Create user message
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    // Loading message
    const loadingMessage: Message = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    // Update or create conversation
    let updatedConversation: Conversation;

    if (currentConversation) {
      updatedConversation = {
        ...currentConversation,
        messages: [...currentConversation.messages, userMessage, loadingMessage],
      };
    } else {
      updatedConversation = {
        threadId: '',
        messages: [userMessage, loadingMessage],
        company: null,
        createdAt: new Date(),
      };
    }

    setCurrentConversation(updatedConversation);

    try {
      let response;

      if (currentConversation?.threadId) {
        // Continue existing conversation
        response = await researchApi.continueConversation(
          currentConversation.threadId,
          content
        );
      } else {
        // New query
        response = await researchApi.query(content, true);
      }

      // Handle interrupt (clarification needed)
      if (response.interrupted && response.interrupt_info) {
        setInterruptInfo(response.interrupt_info);

        // Update conversation without assistant response yet
        const interruptedConversation: Conversation = {
          ...updatedConversation,
          threadId: response.thread_id,
          messages: updatedConversation.messages.filter((m) => !m.isLoading),
        };
        setCurrentConversation(interruptedConversation);

        // Add to conversations if new
        if (!currentConversation) {
          setConversations((prev) => [interruptedConversation, ...prev]);
        } else {
          setConversations((prev) =>
            prev.map((c) =>
              c.threadId === interruptedConversation.threadId
                ? interruptedConversation
                : c
            )
          );
        }

        return;
      }

      // Create assistant response message
      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.final_response || 'No response generated.',
        timestamp: new Date(),
      };

      // Fetch state for additional info
      try {
        const stateResponse = await researchApi.getConversationState(response.thread_id);
        setCurrentState({
          company: stateResponse.state.detected_company,
          attempts: stateResponse.state.research_attempts,
          confidence: stateResponse.state.confidence_score,
          cached: response.cached,
        });
      } catch {
        // State fetch failed, use cached flag
        setCurrentState((prev) => ({ ...prev, cached: response.cached }));
      }

      // Update conversation with response
      const finalConversation: Conversation = {
        ...updatedConversation,
        threadId: response.thread_id,
        messages: [
          ...updatedConversation.messages.filter((m) => !m.isLoading),
          assistantMessage,
        ],
        company: currentState.company,
      };

      setCurrentConversation(finalConversation);

      // Add to conversations list
      if (!currentConversation) {
        setConversations((prev) => [finalConversation, ...prev]);
      } else {
        setConversations((prev) =>
          prev.map((c) =>
            c.threadId === finalConversation.threadId ? finalConversation : c
          )
        );
      }

      // Update cache stats
      const newCacheStats = await researchApi.getCacheStats();
      setCacheStats(newCacheStats);
    } catch (error: any) {
      console.error('Error:', error);

      // Error message
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Error: ${error.message || 'Failed to process your request. Please try again.'}`,
        timestamp: new Date(),
      };

      const errorConversation: Conversation = {
        ...updatedConversation,
        messages: [
          ...updatedConversation.messages.filter((m) => !m.isLoading),
          errorMessage,
        ],
      };

      setCurrentConversation(errorConversation);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle clarification response
  const handleClarification = async (clarification: string) => {
    if (!currentConversation?.threadId || !interruptInfo) return;

    setIsLoading(true);
    setInterruptInfo(null);

    // Add clarification as user message
    const clarificationMessage: Message = {
      id: generateId(),
      role: 'user',
      content: clarification,
      timestamp: new Date(),
    };

    const loadingMessage: Message = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    const updatedConversation: Conversation = {
      ...currentConversation,
      messages: [...currentConversation.messages, clarificationMessage, loadingMessage],
    };

    setCurrentConversation(updatedConversation);

    try {
      const response = await researchApi.clarify(
        currentConversation.threadId,
        clarification
      );

      // Handle another interrupt
      if (response.interrupted && response.interrupt_info) {
        setInterruptInfo(response.interrupt_info);

        const interruptedConversation: Conversation = {
          ...updatedConversation,
          messages: updatedConversation.messages.filter((m) => !m.isLoading),
        };
        setCurrentConversation(interruptedConversation);
        return;
      }

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.final_response || 'No response generated.',
        timestamp: new Date(),
      };

      const finalConversation: Conversation = {
        ...updatedConversation,
        messages: [
          ...updatedConversation.messages.filter((m) => !m.isLoading),
          assistantMessage,
        ],
      };

      setCurrentConversation(finalConversation);
      setConversations((prev) =>
        prev.map((c) =>
          c.threadId === finalConversation.threadId ? finalConversation : c
        )
      );
    } catch (error: any) {
      console.error('Clarification error:', error);

      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Error: ${error.message || 'Failed to process clarification.'}`,
        timestamp: new Date(),
      };

      const errorConversation: Conversation = {
        ...updatedConversation,
        messages: [
          ...updatedConversation.messages.filter((m) => !m.isLoading),
          errorMessage,
        ],
      };

      setCurrentConversation(errorConversation);
    } finally {
      setIsLoading(false);
    }
  };

  // Cancel clarification
  const handleCancelClarification = () => {
    setInterruptInfo(null);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <Sidebar
        companies={companies}
        conversations={conversations}
        currentThreadId={currentConversation?.threadId || null}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        onSelectCompany={handleSelectCompany}
        cacheStats={cacheStats}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Status Bar */}
        <StatusBar
          isConnected={isConnected}
          currentCompany={currentState.company}
          researchAttempts={currentState.attempts}
          confidenceScore={currentState.confidence}
          cached={currentState.cached}
        />

        {/* Chat Area */}
        {currentConversation ? (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto">
              {currentConversation.messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <ChatInput
              onSend={handleSendMessage}
              isLoading={isLoading}
              placeholder="Ask a follow-up question..."
            />
          </>
        ) : (
          <>
            {/* Welcome Screen */}
            <WelcomeScreen
              onQuickQuery={handleSendMessage}
              companies={companies}
            />

            {/* Input */}
            <ChatInput
              onSend={handleSendMessage}
              isLoading={isLoading}
              suggestedQueries={[
                "What's Apple working on?",
                'Tell me about NVIDIA',
                'Microsoft AI strategy',
              ]}
            />
          </>
        )}
      </div>

      {/* Clarification Modal */}
      {interruptInfo && (
        <ClarificationModal
          question={interruptInfo.question}
          originalQuery={interruptInfo.original_query}
          onSubmit={handleClarification}
          onCancel={handleCancelClarification}
        />
      )}
    </div>
  );
}

export default App;
