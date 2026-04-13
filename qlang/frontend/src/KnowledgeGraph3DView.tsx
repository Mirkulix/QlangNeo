import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'

type GraphType = 'Chat' | 'GoalExecution' | 'AgentTask' | 'Evolution' | 'ValueCheck'

type StoredNode = {
  id: string
  label: string
  op: string
  node_type: 'Input' | 'Output' | 'Llm' | 'Deterministic' | 'Memory' | 'Values'
  agent?: string | null
  status?: 'Pending' | 'Running' | 'Completed' | 'Failed'
}

type StoredEdge = {
  from: string
  to: string
  data_type?: string
}

type StoredGraph = {
  id: number
  timestamp: number
  graph_type: GraphType
  title: string
  nodes: StoredNode[]
  edges: StoredEdge[]
}

type GraphStats = {
  total_graphs: number
  by_type: Record<string, number>
}

type MessageEvent = {
  id: number
  from: string
  to: string
  intent: string
  graph_name: string
  timestamp: number
}

type UiNode = {
  id: string
  label: string
  kind: string
  size: number
  val: number
  color: string
  sourceGraph: string
}

type UiLink = {
  source: string
  target: string
  label: string
  live?: boolean
}

function colorForKind(kind: string): string {
  switch (kind) {
    case 'Llm':
      return '#3b82f6'
    case 'Memory':
      return '#22c55e'
    case 'Deterministic':
      return '#f59e0b'
    case 'Values':
      return '#a855f7'
    case 'Input':
      return '#64748b'
    case 'Output':
      return '#e11d48'
    case 'Agent':
      return '#06b6d4'
    default:
      return '#94a3b8'
  }
}

function extractTerms(text: string, maxTerms = 10): string[] {
  const stopWords = new Set([
    'und', 'oder', 'aber', 'dass', 'der', 'die', 'das', 'ein', 'eine', 'mit', 'von', 'fuer',
    'für', 'ist', 'sind', 'was', 'wie', 'ich', 'du', 'wir', 'sie', 'the', 'and', 'for', 'with',
    'this', 'that', 'from', 'into', 'also', 'not', 'have', 'has', 'will', 'can', 'your', 'you'
  ])

  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9äöüß\s-]/gi, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 4 && !stopWords.has(t))

  const freq = new Map<string, number>()
  for (const token of tokens) {
    freq.set(token, (freq.get(token) ?? 0) + 1)
  }

  return Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxTerms)
    .map(([t]) => t)
}

export default function KnowledgeGraph3DView() {
  const fgRef = useRef<any>(null)

  const [graphs, setGraphs] = useState<StoredGraph[]>([])
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [liveLinks, setLiveLinks] = useState<UiLink[]>([])

  const [sourceName, setSourceName] = useState('KnowHow Input')
  const [knowHow, setKnowHow] = useState('')
  const [ideas, setIdeas] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [status, setStatus] = useState('')

  const refresh = async () => {
    try {
      const [gRes, sRes] = await Promise.all([
        fetch('/api/graphs?limit=200'),
        fetch('/api/graphs/stats'),
      ])
      if (gRes.ok) {
        const data = await gRes.json()
        setGraphs(Array.isArray(data) ? data : [])
      }
      if (sRes.ok) {
        const s = await sRes.json()
        setStats(s)
      }
    } catch {
      setStatus('Backend nicht erreichbar, verwende lokale Ansicht')
    }
  }

  useEffect(() => {
    refresh()
    const id = window.setInterval(refresh, 5000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    const es = new EventSource('/api/messages/stream')
    es.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data) as MessageEvent
        const link: UiLink = {
          source: `agent:${msg.from}`,
          target: `agent:${msg.to}`,
          label: msg.intent,
          live: true,
        }
        setLiveLinks((prev) => {
          const next = [link, ...prev]
          return next.slice(0, 40)
        })
      } catch {
        // ignore malformed events
      }
    }
    es.onerror = () => {
      setStatus('Live-Stream getrennt, versuche Reconnect ...')
    }
    return () => es.close()
  }, [])

  const graphData = useMemo(() => {
    const nodeMap = new Map<string, UiNode>()
    const links: UiLink[] = []

    for (const g of graphs) {
      for (const n of g.nodes) {
        const id = `g${g.id}:${n.id}`
        if (!nodeMap.has(id)) {
          nodeMap.set(id, {
            id,
            label: n.label,
            kind: n.node_type,
            size: 5,
            val: 3,
            color: colorForKind(n.node_type),
            sourceGraph: g.title,
          })
        }

        if (n.agent) {
          const aId = `agent:${n.agent}`
          if (!nodeMap.has(aId)) {
            nodeMap.set(aId, {
              id: aId,
              label: n.agent,
              kind: 'Agent',
              size: 7,
              val: 5,
              color: colorForKind('Agent'),
              sourceGraph: 'agent-stream',
            })
          }
          links.push({ source: aId, target: id, label: 'works_on' })
        }
      }

      for (const e of g.edges) {
        links.push({
          source: `g${g.id}:${e.from}`,
          target: `g${g.id}:${e.to}`,
          label: e.data_type ?? 'flow',
        })
      }
    }

    for (const l of liveLinks) {
      if (!nodeMap.has(l.source)) {
        nodeMap.set(l.source, {
          id: l.source,
          label: l.source.replace('agent:', ''),
          kind: 'Agent',
          size: 7,
          val: 5,
          color: colorForKind('Agent'),
          sourceGraph: 'agent-stream',
        })
      }
      if (!nodeMap.has(l.target)) {
        nodeMap.set(l.target, {
          id: l.target,
          label: l.target.replace('agent:', ''),
          kind: 'Agent',
          size: 7,
          val: 5,
          color: colorForKind('Agent'),
          sourceGraph: 'agent-stream',
        })
      }
      links.push(l)
    }

    return {
      nodes: Array.from(nodeMap.values()),
      links,
    }
  }, [graphs, liveLinks])

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge').strength(-140)
      fgRef.current.d3Force('link').distance(65)
    }
  }, [graphData.nodes.length])

  const ingestKnowHow = async () => {
    if (!knowHow.trim() && !ideas.trim()) {
      setStatus('Bitte Know-how oder Ideen eingeben')
      return
    }

    setIngesting(true)
    setStatus('Ingestion laeuft ...')

    try {
      const combined = `${knowHow}\n${ideas}`
      const terms = extractTerms(combined, 12)
      const now = Math.floor(Date.now() / 1000)
      const graphTitle = `${sourceName} @ ${new Date().toLocaleString()}`

      const nodes: StoredNode[] = [
        {
          id: 'source',
          op: 'ingest',
          node_type: 'Memory',
          label: sourceName,
          status: 'Completed',
        },
        {
          id: 'llm-router',
          op: 'reasoning',
          node_type: 'Llm',
          label: 'Multi-LLM Router',
          status: 'Completed',
        },
      ]

      const edges: StoredEdge[] = [
        { from: 'source', to: 'llm-router', data_type: 'knowhow' },
      ]

      for (let i = 0; i < terms.length; i++) {
        const term = terms[i]
        const termId = `idea-${i}`
        nodes.push({
          id: termId,
          op: 'concept',
          node_type: 'Memory',
          label: term,
          status: 'Completed',
        })
        edges.push({ from: 'llm-router', to: termId, data_type: 'extracts' })
      }

      const payload = {
        id: 0,
        timestamp: now,
        graph_type: 'AgentTask',
        title: graphTitle,
        nodes,
        edges,
        metadata: {
          llm_tier: 'hybrid',
          tokens_estimated: combined.length,
        },
      }

      const res = await fetch('/api/graphs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(txt || 'POST /api/graphs fehlgeschlagen')
      }

      setStatus('Know-how erfolgreich in den Graph ingestiert')
      setKnowHow('')
      setIdeas('')
      await refresh()
    } catch (e) {
      setStatus(`Ingestion Fehler: ${e instanceof Error ? e.message : 'unbekannt'}`)
    } finally {
      setIngesting(false)
    }
  }

  return (
    <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 12 }}>
        <div className="card-static" style={{ padding: 12 }}>
          <h3 style={{ marginBottom: 8 }}>Know-how Ingestion</h3>
          <input
            className="input"
            placeholder="Quelle (z.B. Kundenprojekt, Forschungsnotiz, Strategie)"
            value={sourceName}
            onChange={(e) => setSourceName(e.target.value)}
            style={{ marginBottom: 8 }}
          />
          <textarea
            className="input"
            placeholder="Dein Datenwissen / Know-how"
            value={knowHow}
            onChange={(e) => setKnowHow(e.target.value)}
            style={{ minHeight: 84, marginBottom: 8 }}
          />
          <textarea
            className="input"
            placeholder="Deine Ideen / Hypothesen"
            value={ideas}
            onChange={(e) => setIdeas(e.target.value)}
            style={{ minHeight: 84, marginBottom: 8 }}
          />
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary" onClick={ingestKnowHow} disabled={ingesting}>
              {ingesting ? 'Ingesting ...' : 'In Graph speichern'}
            </button>
            <button className="btn" onClick={refresh}>Refresh</button>
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-secondary)' }}>{status}</div>
        </div>

        <div className="card-static" style={{ padding: 12 }}>
          <h3 style={{ marginBottom: 8 }}>Live Status</h3>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', display: 'grid', gap: 6 }}>
            <div>Graphs total: <strong>{stats?.total_graphs ?? graphs.length}</strong></div>
            <div>Nodes (current view): <strong>{graphData.nodes.length}</strong></div>
            <div>Links (current view): <strong>{graphData.links.length}</strong></div>
            <div>Live agent links: <strong>{liveLinks.length}</strong></div>
            <div style={{ marginTop: 4 }}>
              {stats?.by_type && Object.keys(stats.by_type).length > 0
                ? Object.entries(stats.by_type).map(([k, v]) => (
                    <span key={k} className="badge" style={{ marginRight: 6, marginBottom: 6 }}>{k}: {v}</span>
                  ))
                : <span className="label">Noch keine Typ-Statistik</span>}
            </div>
          </div>
        </div>
      </div>

      <div className="card-static" style={{ flex: 1, minHeight: 480, padding: 8 }}>
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData as any}
          nodeLabel={(n: any) => `${n.label} [${n.kind}]\nsource: ${n.sourceGraph}`}
          nodeColor={(n: any) => n.color}
          nodeVal={(n: any) => n.val}
          linkColor={(l: any) => (l.live ? '#22c55e' : 'rgba(148,163,184,0.55)')}
          linkDirectionalParticles={(l: any) => (l.live ? 3 : 0)}
          linkDirectionalParticleWidth={(l: any) => (l.live ? 2.5 : 0)}
          backgroundColor="#06080f"
          onNodeClick={(node: any) => {
            const dist = 120
            const ratio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1)
            fgRef.current?.cameraPosition(
              {
                x: (node.x || 1) * ratio,
                y: (node.y || 1) * ratio,
                z: (node.z || 1) * ratio,
              },
              node,
              800
            )
          }}
        />
      </div>
    </div>
  )
}
