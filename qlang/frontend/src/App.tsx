import { useState, useEffect } from 'react'
import {
  MessageSquare,
  Target,
  Users,
  Brain,
  Server,
  Dna,
  GitBranch,
  Clock,
  Mail,
  Zap,
  Sun,
  Moon,
  Sparkles,
} from 'lucide-react'
import NeoShell from './neo/NeoShell'
import ChatView from './ChatView'
import ConsciousnessView from './ConsciousnessView'
import GoalsView from './GoalsView'
import AgentsView from './AgentsView'
import EvolutionView from './EvolutionView'
import EvolutionStrategyView from './EvolutionStrategyView'
import GraphsView from './GraphsView'
import HistorieView from './HistorieView'
import ProviderView from './ProviderView'
import MessagesView from './MessagesView'
import TrainingView from './TrainingView'
import GpuTrainingView from './GpuTrainingView'
import SpikingView from './SpikingView'
import OrganismView from './OrganismView'
import ActivityFeed from './ActivityFeed'
import NeoAgents from './neo/NeoAgents'
import KnowledgeGraph3DView from './KnowledgeGraph3DView'

type Tab = 'neo' | 'chat' | 'goals' | 'agents' | 'consciousness' | 'provider' | 'evolution' | 'strategy' | 'graphs' | 'knowledge-3d' | 'messages' | 'training' | 'gpu-training' | 'spiking' | 'organism' | 'historie' | 'neo-agents'

const tabs: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: 'neo', label: 'Neo', icon: Sparkles },
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'goals', label: 'Ziele', icon: Target },
  { id: 'agents', label: 'Agenten', icon: Users },
  { id: 'consciousness', label: 'Bewusstsein', icon: Brain },
  { id: 'provider', label: 'Provider', icon: Server },
  { id: 'evolution', label: 'Evolution', icon: Dna },
  { id: 'strategy', label: 'Strategie', icon: GitBranch },
  { id: 'graphs', label: 'QLANG', icon: GitBranch },
  { id: 'knowledge-3d', label: 'Knowledge 3D', icon: Brain },
  { id: 'messages', label: 'Messages', icon: Mail },
  { id: 'training', label: 'Training', icon: Dna },
  { id: 'gpu-training', label: 'GPU Training', icon: Zap },
  { id: 'spiking', label: 'Spiking', icon: Brain },
  { id: 'organism', label: 'Organismus', icon: Brain },
  { id: 'historie', label: 'Historie', icon: Clock },
  { id: 'neo-agents', label: 'Neo Agents', icon: Users },
]

function renderView(tab: Tab, onNavigate: (tab: string) => void) {
  switch (tab) {
    case 'neo': return <NeoShell />
    case 'chat': return <ChatView />
    case 'goals': return <GoalsView />
    case 'agents': return <AgentsView />
    case 'consciousness': return <ConsciousnessView />
    case 'provider': return <ProviderView />
    case 'evolution': return <EvolutionView />
    case 'strategy': return <EvolutionStrategyView />
    case 'graphs': return <GraphsView />
    case 'knowledge-3d': return <KnowledgeGraph3DView />
    case 'messages': return <MessagesView />
    case 'training': return <TrainingView />
    case 'gpu-training': return <GpuTrainingView />
    case 'spiking': return <SpikingView />
    case 'organism': return <OrganismView />
    case 'historie': return <HistorieView onNavigate={onNavigate} />
    case 'neo-agents': return <NeoAgents />
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('neo')
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('qo-theme')
    return saved === 'dark'
  })
  const [connected, setConnected] = useState<boolean>(true)

  // Apply theme to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.setAttribute('data-theme', 'dark')
      localStorage.setItem('qo-theme', 'dark')
    } else {
      document.documentElement.removeAttribute('data-theme')
      localStorage.setItem('qo-theme', 'light')
    }
  }, [darkMode])

  // Poll /api/health every 30 seconds for connection status
  useEffect(() => {
    const checkHealth = () => {
      fetch('/api/health')
        .then(r => setConnected(r.ok))
        .catch(() => setConnected(false))
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30_000)
    return () => clearInterval(interval)
  }, [])

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey

      // Ctrl+1..8 — switch tabs
      if (ctrl && e.key >= '1' && e.key <= '8') {
        const idx = parseInt(e.key, 10) - 1
        if (idx < tabs.length) {
          e.preventDefault()
          setActiveTab(tabs[idx].id)
        }
        return
      }

      // Ctrl+K — focus chat input
      if (ctrl && e.key === 'k') {
        e.preventDefault()
        setActiveTab('chat')
        // ChatView listens to a custom event to focus its input
        window.dispatchEvent(new CustomEvent('qo:focus-chat-input'))
        return
      }

      // Escape — blur / close (dispatch event for consumers)
      if (e.key === 'Escape') {
        window.dispatchEvent(new CustomEvent('qo:escape'))
        return
      }

      // Ctrl+Enter / Cmd+Enter — send chat message
      if (ctrl && e.key === 'Enter') {
        if (activeTab === 'chat') {
          e.preventDefault()
          window.dispatchEvent(new CustomEvent('qo:send-chat'))
        }
        return
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeTab])

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>QO</h1>
        </div>
        <nav className="sidebar-nav">
          {tabs.map(tab => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                className={`nav-item${activeTab === tab.id ? ' active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon className="nav-icon" size={20} />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer" style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'center' }}>
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setDarkMode(d => !d)}
            title={darkMode ? 'Light Mode' : 'Dark Mode'}
            aria-label={darkMode ? 'Light Mode aktivieren' : 'Dark Mode aktivieren'}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <div
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            title={connected ? 'Verbunden' : 'Nicht verbunden'}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: connected ? '#22c55e' : '#ef4444',
              display: 'inline-block',
              flexShrink: 0,
            }} />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {connected ? 'Online' : 'Offline'}
            </span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>v0.1.0</span>
        </div>
      </aside>

      <div className="main-area">
        <div className="main-content">
          {renderView(activeTab, (t) => setActiveTab(t as Tab))}
        </div>
        <div className="activity-panel">
          <ActivityFeed />
        </div>
      </div>
    </div>
  )
}
